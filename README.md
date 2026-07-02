# Codebase RAG Assistant

Paste a GitHub repo URL, let it index the code, then ask questions in plain English — every answer is grounded in the actual source and **cites file paths and line numbers** you can check. The same retrieval pattern that powers Cursor and Sourcegraph Cody, stripped to its shippable core.

**Live demo:** _coming soon — deploy in progress_ <!-- TODO: add Vercel URL -->

## How it works

```
GitHub URL ──> clone (GitPython) ──> language-aware chunking ──> BGE embeddings (384-d)
                                                                       │
                                                                       ▼
User question ──> embed query ──> pgvector cosine search ──────> Postgres (Supabase)
                                        │
                                        ▼
                        top-6 chunks + file:line labels ──> Gemini ──> cited answer
```

1. **Ingest** — the repo is shallow-cloned, non-code files are skipped, and each code file is split with LangChain's `RecursiveCharacterTextSplitter.from_language`, which respects function/class boundaries per language. Every chunk keeps its file path and start/end line numbers.
2. **Embed** — chunks are embedded locally with `BAAI/bge-small-en-v1.5` (free, no API key, 384 dimensions).
3. **Store** — vectors live in Postgres via **pgvector**, next to the chunk metadata, with an IVFFlat cosine index.
4. **Answer** — a question is embedded (with the BGE query instruction), the top-6 chunks are retrieved by cosine distance, and Gemini is prompted to answer *only* from that context, citing `file_path:start-end` for every claim.

## Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI (Python) |
| Vectors + metadata | Postgres + pgvector (Supabase) |
| Embeddings | sentence-transformers `BAAI/bge-small-en-v1.5` |
| Generation | Gemini API |
| Frontend | React (Vite) |
| Deploy | Render (backend) + Vercel (frontend) |

## Run it locally

Prereqs: Python 3.11+, Node 18+, a Postgres with pgvector (free [Supabase](https://supabase.com) project works), a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
# backend
cd backend
python -m venv venv && venv/Scripts/activate   # or source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                            # fill in DATABASE_URL + GEMINI_API_KEY
uvicorn app.main:app --reload                   # http://localhost:8000/docs

# frontend (separate terminal)
cd frontend
npm install
npm run dev                                     # http://localhost:5173
```

No Supabase? `docker compose up -d` starts a local pgvector Postgres; point `DATABASE_URL` at it.

## API

- `POST /index` `{"github_url": "..."}` → `{"repo", "chunk_count"}` — clone, chunk, embed, store (re-indexing replaces old chunks)
- `POST /query` `{"question": "...", "repo": "..."}` → `{"answer", "sources": [{file_path, start_line, end_line, ...}]}`
- `GET /repos` — list indexed repos
- `GET /health`

Interactive Swagger docs at `/docs`.

## Design decisions

- **pgvector over a dedicated vector DB** — one datastore holds both relational metadata and vectors, so inserts are transactional and ops stay trivial. Qdrant/Weaviate earn their complexity at much larger scale than v1 needs.
- **Language-aware chunking over fixed-size cuts** — splitting on function/class boundaries keeps each chunk semantically whole, so its embedding actually represents one idea. Arbitrary character cuts split functions mid-body and blur retrieval.
- **BGE-small locally over an embeddings API** — free, fast (384-dim), no key management, and a top open model on MTEB. Indexing a mid-size repo takes seconds on CPU.
- **RAG over stuffing the repo in the prompt** — repos exceed context windows, tokens cost money, and retrieval focuses the model on relevant code. Grounding in retrieved chunks (with citations) is also what makes answers *checkable* instead of plausible-sounding.

## Future work

- Hybrid retrieval (BM25 + vector) with a reranker
- Token-by-token streaming responses over WebSockets
- Evaluation harness: labeled question set + RAGAS metrics (faithfulness, answer relevance)
- Auth (JWT), per-user repos, persisted chat history
- Code dependency/call-graph retrieval (networkx → Neo4j at scale)
- CI (GitHub Actions) + Docker Compose polish
