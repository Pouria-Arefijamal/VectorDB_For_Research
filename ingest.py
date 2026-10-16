#!/usr/bin/env python3
"""Build a vector database from a folder of PDFs and (optionally) a folder of .md summaries.

Two Chroma collections are created:
  pages      one entry per ~chunk of a PDF page   -> exact page numbers
  summaries  one entry per "## " section of a .md -> dataset, results, references...

Example:
  python ingest.py --pdfs /path/PDFs --summaries /path/Summaries --db ./db \
                   --batch-size 64 --device cuda
"""
import argparse
import re
import sys
from pathlib import Path

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdfs", required=True, type=Path, help="folder containing the PDFs (searched recursively)")
    p.add_argument("--summaries", type=Path, default=None, help="folder containing the .md summaries (optional)")
    p.add_argument("--db", type=Path, default=Path("db"), help="where the database is stored (default: ./db)")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="sentence-transformers model name")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ... (default: auto)")
    p.add_argument("--batch-size", type=int, default=64, help="chunks embedded per GPU/CPU pass (default: 64)")
    p.add_argument("--db-batch", type=int, default=500, help="chunks written to the database per call (default: 500)")
    p.add_argument("--chunk-words", type=int, default=300, help="words per page chunk (default: 300)")
    p.add_argument("--overlap", type=int, default=50, help="overlapping words between chunks (default: 50)")
    p.add_argument("--id-regex", default=r"^([A-Za-z]+\d+)",
                   help="regex whose first group extracts the paper ID from a file name; "
                        "PDF and summary are matched on it (default: leading letters+digits, e.g. P001). "
                        "If it does not match, the whole file stem is used.")
    p.add_argument("--rebuild", action="store_true", help="delete the existing database and start again")
    p.add_argument("--max-papers", type=int, default=None, help="only process the first N PDFs (for testing)")
    return p.parse_args()


def paper_id(path: Path, rx: re.Pattern) -> str:
    m = rx.match(path.stem)
    return m.group(1) if m else path.stem


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def chunk_words(text: str, size: int, overlap: int):
    words = text.split()
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) >= 15:  # ignore headers, page numbers, empty scans
            yield " ".join(piece)
        if start + size >= len(words):
            break


def meta_field(md: str, label: str) -> str:
    m = re.search(rf"\*\*{label}:\*\*\s*(.+)", md)
    return re.sub(r"\s*\((?:Pages?|Sec)[^)]*\)\.?\s*$", "", m.group(1).strip()) if m else ""


def part_key(heading: str) -> str:
    """'PART 5 — DATASET' -> 'dataset';  'PART 18 — REFERENCES (most important)' -> 'references'."""
    h = re.sub(r"^PART\s*\d+\s*[—–-]\s*", "", heading.strip(), flags=re.I)
    h = re.sub(r"\(.*?\)", "", h)
    return h.strip().lower()


def split_summary(md: str):
    """Yield (heading, body) for every '## ' section; text before the first '##' is the 'header'."""
    parts = re.split(r"^##\s+", md, flags=re.M)
    if parts[0].strip():
        yield "header", parts[0].strip()
    for block in parts[1:]:
        heading, _, body = block.partition("\n")
        yield heading.strip(), body.strip()


def flush(col, ids, docs, metas, model, batch_size, db_batch):
    if not ids:
        return
    vecs = model.encode(docs, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    for i in range(0, len(ids), db_batch):
        col.upsert(ids=ids[i:i + db_batch], documents=docs[i:i + db_batch],
                   metadatas=metas[i:i + db_batch], embeddings=vecs[i:i + db_batch].tolist())


def main():
    a = parse_args()
    rx = re.compile(a.id_regex)
    device = resolve_device(a.device)

    pdfs = sorted(a.pdfs.rglob("*.pdf"))
    if a.max_papers:
        pdfs = pdfs[:a.max_papers]
    if not pdfs:
        sys.exit(f"No PDFs found in {a.pdfs}")

    summaries = {}
    if a.summaries:
        for f in sorted(a.summaries.rglob("*.md")):
            summaries[paper_id(f, rx)] = f
        print(f"Found {len(pdfs)} PDFs and {len(summaries)} summaries")
    else:
        print(f"Found {len(pdfs)} PDFs (no summaries folder given)")

    print(f"Loading {a.model} on {device} (batch size {a.batch_size})")
    model = SentenceTransformer(a.model, device=device)

    client = chromadb.PersistentClient(path=str(a.db))
    if a.rebuild:
        for name in ("pages", "summaries"):
            try:
                client.delete_collection(name)
            except Exception:
                pass
    cfg = {"hnsw:space": "cosine", "model": a.model}
    pages = client.get_or_create_collection("pages", metadata=cfg)
    sums = client.get_or_create_collection("summaries", metadata=cfg)
    stored_model = pages.metadata.get("model", a.model)
    if stored_model != a.model:
        sys.exit(f"Database was built with '{stored_model}', not '{a.model}'. Use --rebuild to switch models.")

    done = {m["paper_id"] for m in pages.get(include=["metadatas"])["metadatas"]}
    todo_summ = {m["paper_id"] for m in sums.get(include=["metadatas"])["metadatas"]}

    # ---- summaries -------------------------------------------------------
    info = {}  # paper_id -> title/year/doi, copied onto the page chunks
    ids, docs, metas = [], [], []
    for pid, f in summaries.items():
        md = f.read_text(encoding="utf-8", errors="ignore")
        info[pid] = {"title": meta_field(md, "Title"), "year": meta_field(md, "Year")[:4],
                     "doi": meta_field(md, "DOI")}
        if pid in todo_summ:
            continue
        for n, (heading, body) in enumerate(split_summary(md)):
            if not body:
                continue
            text = f"{info[pid]['title']} — {heading}\n{body}"
            ids.append(f"{pid}:s{n}")
            docs.append(text)
            metas.append({"paper_id": pid, "part": heading, "part_key": part_key(heading),
                          "file": f.name, **info[pid]})
    flush(sums, ids, docs, metas, model, a.batch_size, a.db_batch)
    print(f"Summaries: stored {len(ids)} sections")

    # ---- pdf pages -------------------------------------------------------
    new = 0
    for k, pdf in enumerate(pdfs, 1):
        pid = paper_id(pdf, rx)
        if pid in done:
            continue
        if pid not in summaries:
            print(f"  note: no summary matched for {pdf.name}")
        ids, docs, metas = [], [], []
        try:
            doc = pymupdf.open(pdf)
        except Exception as e:
            print(f"  skip {pdf.name}: {e}")
            continue
        for pno, page in enumerate(doc, 1):
            for c, chunk in enumerate(chunk_words(page.get_text(), a.chunk_words, a.overlap)):
                ids.append(f"{pid}:p{pno}:c{c}")
                docs.append(chunk)
                metas.append({"paper_id": pid, "page": pno, "file": pdf.name,
                              **info.get(pid, {"title": "", "year": "", "doi": ""})})
        doc.close()
        flush(pages, ids, docs, metas, model, a.batch_size, a.db_batch)
        new += 1
        print(f"[{k}/{len(pdfs)}] {pid}: {len(ids)} chunks")

    print(f"Done. New papers: {new}. Pages collection: {pages.count()} chunks, "
          f"summaries collection: {sums.count()} sections. Database: {a.db}")


if __name__ == "__main__":
    main()
