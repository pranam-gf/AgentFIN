# AgentFIN: GoodFin RAG System

A high-performance, agentic Retrieval-Augmented Generation (RAG) pipeline for CFA Level 3 and financial datasets. Built for accurate, explainable, and extensible financial question answering.

## Features

- **Document Ingestion:**
  - Parse PDFs, Markdown, and text files ( Future work : ANy file type can be used)
  - Extract and tag CFA topics and pathways ( TODOS : Add all the topics ( dataset need ))
  - Generate OpenAI embeddings and store in Pinecone ( ( IMP== Multiligual : Google Emb, LEegal : Voyage ( worth looking after we get first test run) Embed, English : Daobuo Embedding))

- **Retrieval Pipeline:**
  - Fast vector search with optional metadata filtering (by topic/pathway) ( DEPENDS ON THE DATASET)
  - Query expansion and multi-stage reranking (configurable) ( STANFORD PAPER CONCEPT )

- **Generation Pipeline:**
  - Assemble context and generate answers using LLMs (OpenAI, etc.) ( TODO : OpenROuter CONFIG FUTURE )
  - Handles long context with chunk batching and synthesis ( dataset needed for additional testing)

- **Extensible:**
  - Modular pipeline (ingestion, retrieval, generation)
  - Easy to add new models, rerankers, or data sources ( TODO )

## Quickstart

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your API keys and settings:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME`
- (Optional) adjust model names, chunk sizes, etc.

### 3. Ingest Documents

```bash
python scripts/ingest_data.py path/to/document.pdf --topic asset_allocation
python scripts/ingest_data.py path/to/folder/ --auto-detect
```

### 4. Query the System

```bash
python scripts/query.py "What are the key considerations for asset allocation?" --topic asset_allocation
python scripts/ask.py "Explain portfolio construction for private wealth clients." --show-sources
```

## Technologies Used

- Python 3.10+
- OpenAI API (embeddings, LLM)
- Pinecone (vector DB)
- LangChain (chunking)
- dotenv, argparse, sentence-transformers (reranking)



