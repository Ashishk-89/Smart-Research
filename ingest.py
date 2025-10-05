""" Ingest helper: arXiv fetcher (abstracts) and optional PDF download + chunking """

import arxiv
import uuid
from vstore import ingest_documents


def fetch_arxiv(query: str, max_results: int = 50):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for res in search.results():
        pid = res.entry_id.split('/')[-1] if res.entry_id else str(uuid.uuid4())
        paper = {
            'id': pid,
            'title': res.title,
            'abstract': res.summary.replace('\n', ' ').strip(),
            'url': res.entry_id,
            'authors': [a.name for a in res.authors],
            'published': res.published.isoformat() if res.published else None
        }
        papers.append(paper)
    return papers


def ingest_query(query: str, n: int = 50):
    papers = fetch_arxiv(query, max_results=n)
    ingest_documents(papers)
    return len(papers)
