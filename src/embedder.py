"""
embedder.py
-----------
Phase 2 — Embed documents and persist them to a ChromaDB vector store.

Uses FastEmbedEmbeddings (BAAI/bge-small-en-v1.5) via ONNX runtime instead
of sentence-transformers/PyTorch. This avoids dependency conflicts with the
system's Python 3.10.12 pyenv build, which is missing the _lzma C extension
required by torchvision.

Two entry points:
  - build_vectorstore : run once to index all 700 documents.
  - load_vectorstore  : call at runtime to reuse the persisted index.
"""

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from src.data_loader import load_and_clean, make_documents

# ChromaDB persists the index to disk so re-embedding on every startup is avoided.
VECTORSTORE_PATH = "vectorstore/chroma_db"

# bge-small-en-v1.5 is compact (33M params) and performs well on
# factual/numeric retrieval tasks like financial Q&A.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings() -> FastEmbedEmbeddings:
    """Return a reusable embedding instance (model is loaded once per process)."""
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore() -> Chroma:
    """
    Load and embed all documents, then persist to ChromaDB.
    Run this once before starting the RAG chain.
    Re-running will overwrite the existing collection.
    """
    df = load_and_clean()
    docs = make_documents(df)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=VECTORSTORE_PATH,
    )
    print(f"Indexed {len(docs)} documents into ChromaDB at '{VECTORSTORE_PATH}'.")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load the persisted ChromaDB index for use in retrieval."""
    return Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=get_embeddings(),
    )


if __name__ == "__main__":
    build_vectorstore()
