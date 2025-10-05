""" Vector store helpers using LangChain abstractions + Chroma """

import os
from dotenv import load_dotenv
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

load_dotenv()

CHROMA_DIR = os.getenv('CHROMA_PERSIST_DIR', './data/chroma_db')
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)
    return _embeddings


def init_or_get_chroma(collection_name: str = 'papers') -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
        embedding_function=embeddings
    )

def clean_metadata(meta: dict) -> dict:
    """Flatten metadata so Chroma accepts it (no lists, only str/int/float/bool/None)."""
    cleaned = {}
    for k, v in meta.items():
        if isinstance(v, list):
            cleaned[k] = ", ".join(map(str, v))   # convert list to comma-separated string
        elif isinstance(v, (str, int, float, bool)) or v is None:
            cleaned[k] = v
        else:
            cleaned[k] = str(v)  # fallback, stringify anything unexpected
    return cleaned

def ingest_documents(docs: list, collection_name: str = 'papers'):
    """
    docs: list of dicts with keys: id, title, abstract, url, authors, published
    """
    vectordb = init_or_get_chroma(collection_name)
    lc_docs = []
    for d in docs:
        md = {
            "title": d.get("title"),
            "url": d.get("url"),
            "authors": d.get("authors"),   # may be a list
            "published": d.get("published"),
            "source_id": d.get("id"),
        }
        cleaned_md = clean_metadata(md)
        lc_docs.append(Document(page_content=d.get("abstract", ""), metadata=cleaned_md))

    vectordb.add_documents(lc_docs)
    vectordb.persist()


def search(query: str, k: int = 5, collection_name: str = 'papers'):
    vectordb = init_or_get_chroma(collection_name)
    results = vectordb.similarity_search_with_score(query, k=k)
    out = []
    for doc, score in results:
        meta = doc.metadata or {}
        out.append({
            'title': meta.get('title', 'unknown'),
            'url': meta.get('url', ''),
            'abstract': doc.page_content,
            'score': score,
            'source_id': meta.get('source_id')
        })
    return out
