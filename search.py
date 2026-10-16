#!/usr/bin/env python3
"""Search the database built by ingest.py.

Examples:
  python search.py "which dataset does the paper use" --paper P012 --kind summaries
  python search.py "how is latency measured" --kind pages -k 8
  python search.py --paper P001 --part dataset          # print that summary section directly
  python search.py "traffic steering" --year 2024
  python search.py --list-parts
"""
import argparse
import sys
from pathlib import Path

import chromadb


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", nargs="?", help="what to look for (optional if --paper and --part are given)")
    p.add_argument("--db", type=Path, default=Path("db"))
    p.add_argument("--kind", choices=["pages", "summaries", "both"], default="both")
    p.add_argument("-k", type=int, default=5, help="results per collection (default: 5)")
    p.add_argument("--paper", help="restrict to one paper ID, e.g. P012")
    p.add_argument("--part", help="restrict summaries to one section, e.g. dataset, results, references")
    p.add_argument("--year", help="restrict to one publication year")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda (query embedding; cpu is fine)")
    p.add_argument("--width", type=int, default=600, help="characters of text to print (default: 600)")
    p.add_argument("--list-parts", action="store_true", help="show the available summary sections and exit")
    return p.parse_args()


def build_where(a, summaries: bool):
    conds = []
    if a.paper:
        conds.append({"paper_id": a.paper})
    if a.year:
        conds.append({"year": a.year})
    if a.part and summaries:
        conds.append({"part_key": a.part.lower()})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def show(kind, doc, meta, dist, width):
    where = f"page {meta['page']}" if kind == "pages" else f"section: {meta['part']}"
    head = f"{meta['paper_id']} | {meta.get('title') or meta['file']} ({meta.get('year') or 'n.d.'}) | {where}"
    if meta.get("doi"):
        head += f" | DOI {meta['doi']}"
    score = f"  [similarity {1 - dist:.2f}]" if dist is not None else ""
    text = " ".join(doc.split())
    print(f"\n{head}{score}\n  {text[:width]}{'…' if len(text) > width else ''}")


def main():
    a = parse_args()
    client = chromadb.PersistentClient(path=str(a.db))
    names = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    if not {"pages", "summaries"} & names:
        sys.exit(f"No database in {a.db}. Run ingest.py first.")

    if a.list_parts:
        sums = client.get_collection("summaries")
        parts = sorted({m["part_key"] for m in sums.get(include=["metadatas"])["metadatas"]})
        print("\n".join(parts))
        return

    if not a.query and not (a.paper and a.part):
        sys.exit("Give a query, or both --paper and --part.")

    # No query: print the requested summary section(s) exactly as stored.
    if not a.query:
        sums = client.get_collection("summaries")
        got = sums.get(where=build_where(a, True), include=["documents", "metadatas"])
        if not got["ids"]:
            sys.exit("Nothing matched.")
        for doc, meta in zip(got["documents"], got["metadatas"]):
            print(f"\n{meta['paper_id']} | {meta['title']} | {meta['part']}\n{doc}")
        return

    from sentence_transformers import SentenceTransformer
    cols = [n for n in ("pages", "summaries") if a.kind in (n, "both") and n in names]
    model_name = client.get_collection(cols[0]).metadata["model"]
    device = a.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""
    q = model.encode([prefix + a.query], normalize_embeddings=True)[0].tolist()

    for name in cols:
        col = client.get_collection(name)
        res = col.query(query_embeddings=[q], n_results=a.k, where=build_where(a, name == "summaries"),
                        include=["documents", "metadatas", "distances"])
        print(f"\n===== {name} =====")
        if not res["ids"][0]:
            print("(no results)")
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            show(name, doc, meta, dist, a.width)


if __name__ == "__main__":
    main()
