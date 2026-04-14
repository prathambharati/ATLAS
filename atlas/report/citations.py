"""Citation manager — tracks sources and formats inline citations.

Assigns citation numbers [1], [2], etc. to sources and provides
a bibliography section at the end of the report.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class Citation:
    """A single citation entry."""

    number: int
    source: str
    title: str = ""
    url: str = ""
    content_preview: str = ""

    def format_inline(self) -> str:
        """Format as inline citation, e.g. [1]."""
        return f"[{self.number}]"

    def format_bibliography(self) -> str:
        """Format as bibliography entry."""
        parts = [f"[{self.number}]"]
        if self.title:
            parts.append(self.title)
        parts.append(f"Source: {self.source}")
        if self.url:
            parts.append(self.url)
        return " — ".join(parts)


class CitationManager:
    """Track and manage citations across a report."""

    def __init__(self):
        self._citations: dict[str, Citation] = {}
        self._counter = 0

    def add_source(
        self,
        source: str,
        title: str = "",
        url: str = "",
        content_preview: str = "",
    ) -> Citation:
        """Add a source and get its citation.

        If the source was already added, returns the existing citation.
        """
        if source in self._citations:
            return self._citations[source]

        self._counter += 1
        citation = Citation(
            number=self._counter,
            source=source,
            title=title,
            url=url,
            content_preview=content_preview,
        )
        self._citations[source] = citation

        log.info(
            "citation_added",
            number=self._counter,
            source=source[:60],
        )
        return citation

    def get_citation(self, source: str) -> Citation | None:
        """Get citation for a source."""
        return self._citations.get(source)

    def format_bibliography(self) -> str:
        """Generate the full bibliography section."""
        if not self._citations:
            return ""

        lines = ["## References", ""]
        sorted_citations = sorted(
            self._citations.values(), key=lambda c: c.number
        )
        for citation in sorted_citations:
            lines.append(citation.format_bibliography())

        return "\n".join(lines)

    @property
    def num_citations(self) -> int:
        return len(self._citations)

    def to_list(self) -> list[dict]:
        """Export citations as a list of dicts."""
        return [
            {
                "number": c.number,
                "source": c.source,
                "title": c.title,
                "url": c.url,
            }
            for c in sorted(
                self._citations.values(), key=lambda c: c.number
            )
        ]
