"""Arxiv search tool for academic paper retrieval.

Uses the arxiv Python package to search for papers by keyword.
Returns paper titles, abstracts, authors, and PDF links.
No API key required — arxiv's API is free and open.
"""

import arxiv

from atlas.observability.logger import get_logger

log = get_logger(__name__)


class ArxivSearchTool:
    """Search arxiv for academic papers."""

    def __init__(self, max_results_default: int = 5):
        self.max_results_default = max_results_default
        self._client = arxiv.Client()
        log.info("arxiv_search_ready")

    def search(
        self,
        query: str,
        max_results: int | None = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    ) -> list[dict]:
        """Search arxiv for papers matching a query.

        Args:
            query: Search query (supports arxiv search syntax).
            max_results: Number of results to return.
            sort_by: Sort by Relevance, LastUpdatedDate, or SubmittedDate.

        Returns:
            List of dicts with paper metadata and abstract.
        """
        max_results = max_results or self.max_results_default

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
            )

            results = []
            for paper in self._client.results(search):
                results.append({
                    "title": paper.title,
                    "abstract": paper.summary,
                    "authors": [a.name for a in paper.authors],
                    "published": paper.published.isoformat() if paper.published else "",
                    "updated": paper.updated.isoformat() if paper.updated else "",
                    "arxiv_id": paper.entry_id.split("/")[-1],
                    "pdf_url": paper.pdf_url,
                    "categories": paper.categories,
                    "source": "arxiv",
                })

            log.info(
                "arxiv_search_complete",
                query=query[:80],
                num_results=len(results),
            )
            return results

        except Exception as e:
            log.error("arxiv_search_failed", query=query, error=str(e))
            return []

    def search_as_chunks(self, query: str, max_results: int | None = None) -> list[dict]:
        """Search and return results in chunk format for retrieval pipeline.

        Combines title + abstract as the chunk text so it can be
        re-ranked alongside vector store and web search results.
        """
        raw_results = self.search(query, max_results=max_results)

        chunks = []
        for i, paper in enumerate(raw_results):
            authors_str = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors_str += f" et al. ({len(paper['authors'])} authors)"

            text = f"{paper['title']}\n\nAuthors: {authors_str}\n\n{paper['abstract']}"

            chunks.append({
                "chunk_id": f"arxiv_{paper['arxiv_id']}",
                "text": text,
                "score": 1.0 - (i * 0.05),  # Decreasing score by rank
                "source": f"arxiv:{paper['arxiv_id']}",
                "metadata": {
                    "source_type": "arxiv",
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "pdf_url": paper["pdf_url"],
                    "published": paper["published"],
                    "arxiv_id": paper["arxiv_id"],
                },
            })

        return chunks
