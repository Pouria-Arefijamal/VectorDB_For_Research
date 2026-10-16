# Paper Vector Search

Search a folder of research-paper PDFs (and their Markdown summaries) **by meaning**, and get back
the **exact PDF page** and the **reference** (title, year, DOI) for every answer.

Typical questions:

- *Which dataset does paper P012 use?*
- *Which papers evaluate on TeleQnA?*
- *On which page is the latency measured?*

Everything runs locally. No API key and no internet connection are needed after the first model download.

## How it works

```
PDFs ──► read page by page ──► ~300-word chunks ──┐
                                                  ├─► embedding model ─► vectors ─► Chroma DB
.md summaries ──► one chunk per "## " section ────┘        (batches, GPU or CPU)
```

Two collections are stored in one database:

| Collection | One entry is… | Gives you |
|---|---|---|
| `pages` | a chunk of one PDF page | the exact page number |
| `summaries` | one `## ` section of a summary file | structured answers (dataset, results, references) |

Every entry carries `paper_id`, `title`, `year`, `doi` and `file` as metadata, so each result is a ready reference.

### Is it SQLite?

Partly, yes. Chroma keeps text and metadata in a SQLite file (`db/chroma.sqlite3`) and the vectors in
index files in the sub-folders of `db/`. You never touch either directly; use `search.py`. Delete the `db/`
folder to start over.

## Install

```bash
git clone <your-repo-url> && cd <your-repo>
./install.sh            # auto-detects a GPU; or: ./install.sh cpu / ./install.sh cuda
source .venv/bin/activate
```

Requirements: Python 3.9+, about 2 GB of disk for the packages. A GPU is optional.
On Jetson or other ARM boards, install the PyTorch build for your device first, then run `./install.sh`.

## Prepare your files

- **PDFs:** any folder (sub-folders are searched).
- **Summaries (optional):** Markdown files, one per paper. Sections start with `## `. The script reads
  `- **Title:**`, `- **Year:**` and `- **DOI:**` lines when present.
- **Matching:** a PDF and its summary are paired by an ID taken from the file name, by default the leading
  letters and digits. `P001_Some_Title.pdf` and `P001_summary.md` both give `P001`. Use `--id-regex` for other schemes.

## 1. Build the database: `ingest.py`

```bash
python ingest.py --pdfs /path/to/PDFs --summaries /path/to/Summaries --db ./db --device cuda --batch-size 64
```

Running it again only processes **new** papers.

| Option | Default | Meaning |
|---|---|---|
| `--pdfs` | required | Folder of PDFs (searched recursively). |
| `--summaries` | none | Folder of `.md` summaries. Leave out to index PDFs only. |
| `--db` | `./db` | Where the database is stored. |
| `--model` | `BAAI/bge-small-en-v1.5` | Any sentence-transformers model. Changing it needs `--rebuild`. |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `cuda:1`. |
| `--batch-size` | `64` | Chunks embedded in one pass. Larger is faster until memory runs out; lower it to 16–32 on a memory error. |
| `--db-batch` | `500` | Chunks written to the database per call. |
| `--chunk-words` | `300` | Words per page chunk. |
| `--overlap` | `50` | Words shared by neighbouring chunks, so sentences are not cut. |
| `--id-regex` | `^([A-Za-z]+\d+)` | Regex whose first group is the paper ID taken from a file name. If it does not match, the whole file name is the ID. |
| `--rebuild` | off | Delete the existing database first. |
| `--max-papers` | all | Process only the first N PDFs (for testing). |

## 2. Ask questions: `search.py`

```bash
python search.py "your question" [options]
```

| Option | Default | Meaning |
|---|---|---|
| `query` | none | What you are looking for. Optional when `--paper` and `--part` are both given. |
| `--db` | `./db` | Database folder. |
| `--kind` | `both` | Where to search: `pages`, `summaries` or `both` (see below). |
| `-k` | `5` | Number of results per collection. |
| `--paper` | none | Only this paper, e.g. `P012`. |
| `--part` | none | Only this summary section, e.g. `dataset`. Applies to summaries. |
| `--year` | none | Only papers from this year. |
| `--device` | `auto` | Device for embedding the query; `cpu` is fine. |
| `--width` | `600` | Characters of text shown per result. |
| `--list-parts` | off | Print the available section names and exit. |

### `--kind`: what each value does

**`--kind summaries`** searches your Markdown summaries only. Each result is one section of one paper, such as
*Dataset*, *Main results* or *References*. Use it for questions about **what a paper is or does**: which dataset,
which baselines, which metrics, what limitations. Answers are short and precise, and they repeat the page
numbers written in your summary. It is the best choice for *"which dataset does this paper use?"*.

```bash
python search.py "which dataset is used" --paper P012 --kind summaries
python search.py "benchmark for telecom question answering" --kind summaries --part dataset -k 10
```

**`--kind pages`** searches the raw PDF text only. Each result is a passage with its **exact PDF page number**.
Use it to **verify a claim** or to find a number, a formula or a table, and to cite the page.

```bash
python search.py "latency is measured in milliseconds" --kind pages -k 8
python search.py "dataset" --paper P001 --kind pages
```

**`--kind both`** (default) prints both lists, summaries first. A good routine is to find the paper with
`summaries`, then confirm the page with `pages`.

### Print one section exactly as written

Give `--paper` and `--part` without a query to get the stored section, with no searching:

```bash
python search.py --paper P001 --part dataset
python search.py --list-parts          # shows valid names: dataset, results, references, ...
```

### Reading a result

```
P001 | 5G INSTRUCT Forge: An Advanced Data Engineering Pipeline… (2025) | page 8 | DOI 10.1109/TCCN.2024.3516055  [similarity 0.72]
  a dataset named OAI Instruct. This dataset was created by applying our data generation pipeline…
```

- `page 8` is the PDF page, counted from 1. It can differ from the number printed on the page.
- `similarity` runs from 0 to 1. Higher is closer in meaning; it is not a probability of being correct.

## Tips

- Always open the cited page before quoting a number. The search finds the page; you check the content.
- Scanned PDFs without a text layer produce no page chunks. Run OCR on them first.
- Use the same `--model` for building and searching. The database remembers it, so `search.py` does this for you.
- Add papers later by putting the files in the folders and running `ingest.py` again.

## Files

| File | Purpose |
|---|---|
| `ingest.py` | Reads PDFs and summaries and builds the database. |
| `search.py` | Queries the database. |
| `install.sh` | Creates `.venv` and installs the dependencies. |
| `requirements.txt` | Python packages (PyTorch is installed by `install.sh`). |
