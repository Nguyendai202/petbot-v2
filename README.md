# VetBot

A Vietnamese-language RAG chatbot that helps pet owners triage dog/cat symptoms,
get a likely diagnosis, and receive treatment guidance — grounded in a veterinary
reference book, with emergency detection and adaptive follow-up questioning.

## How it works

```
User message
   │
   ▼
1. Emergency check ──────────► immediate first-aid response (no retrieval)
   │ (no emergency)
   ▼
2. Classify ─────────────────► ask for missing basic info (species, age, vaccination)
   │ (enough info)
   ▼
3. Query understanding ──────► chitchat short-circuit, or build retrieval queries
   │
   ▼
4. Coarse retrieve ──────────► hybrid search (dense + sparse, RRF fusion) over candidates
   │
   ▼
5. Disambiguate ─────────────► confident enough to commit to a disease?
   │ no → ask a targeted distinguishing question, loop back to user
   │ yes
   ▼
6. Fine retrieve ────────────► treatment-specific chunks for the committed disease
   │                           (+ cross-reference pass, e.g. Parvo → Carre treatment)
   ▼
7. Diagnose ──────────────────► stream final answer with full context
```

## Stack

- **LLM**: OpenAI GPT-4o-mini (routing/classification) + GPT-4o (diagnosis)
- **Retrieval**: Qdrant hybrid search — dense embeddings (`text-embedding-3-large`) +
  sparse TF vectors, fused with RRF
- **UI**: [Chainlit](https://chainlit.io) with SQLAlchemy-backed chat persistence
- **Tracing**: LangSmith
- **Deployment**: Docker + Railway

## Key design decisions

- **Hybrid retrieval over pure dense search** — disease/drug names are exact
  keywords that dense embeddings can miss; sparse vectors catch them.
- **Coarse-to-fine, 2-pass retrieval** — a broad symptom search narrows to a
  candidate disease, then a filtered pass retrieves precise treatment chunks,
  avoiding context dilution when multiple diseases score similarly.
- **Adaptive disambiguation** — the model evaluates whether retrieved evidence
  is sufficient to commit to a diagnosis, asking a targeted follow-up question
  instead of guessing when symptoms overlap across diseases.
- **Split routing/diagnosis models** — cheaper model for classification and
  query understanding, stronger model only for the final diagnosis, to control
  cost without sacrificing answer quality.

## Project layout

```
app.py                 Chainlit entrypoint (auth, message loop)
vetbot/
  chain.py             Conversation flow: classify → retrieve → disambiguate → diagnose
  prompts.py           System prompts for each stage
  retriever.py          Qdrant hybrid retrieval (dense + sparse, coarse-to-fine)
  ingest.py             PDF → disease-aware chunking → embeddings → Qdrant
ingest_once.py          CLI entrypoint to run ingestion
tests/                  Unit tests for pure logic (no API calls)
```

## Setup

```bash
cp .env.example .env   # fill in OPENAI_API_KEY, QDRANT_URL, QDRANT_KEY, APP_USER, APP_PASSWORD
pip install -r requirements.txt
python ingest_once.py  # one-time: ingest the PDF into Qdrant
chainlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests cover pure logic only (tokenization, emergency detection, disease heading
parsing, sparse vector construction, point formatting) — no live API or DB calls.
