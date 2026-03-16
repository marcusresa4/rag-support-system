import arxiv
from dataclasses import dataclass
from datetime import datetime
import structlog

logger = structlog.get_logger()

@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published_date: datetime
    pdf_url: str
    categories: list[str]

def fetch_paper_by_id(arxiv_id: str) -> ArxivPaper:
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))

    if not results:
        raise ValueError(f"Paper not found: {arxiv_id}")

    paper = results[0]
    logger.info("fetched_paper", arxiv_id=arxiv_id, title=paper.title[:60])

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=paper.title,
        abstract=paper.summary,
        authors=[a.name for a in paper.authors],
        published_date=paper.published,
        pdf_url=paper.pdf_url,
        categories=paper.categories,
    )

def fetch_papers_by_category(
    category: str = "cs.AI",
    max_results: int = 100,
) -> list[ArxivPaper]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in client.results(search):
        paper_id = result.entry_id.split("/")[-1]
        papers.append(ArxivPaper(
            arxiv_id=paper_id,
            title=result.title,
            abstract=result.summary,
            authors=[a.name for a in result.authors],
            published_date=result.published,
            pdf_url=result.pdf_url,
            categories=result.categories,
        ))

    logger.info("batch_fetch_complete", category=category, count=len(papers))
    return papers