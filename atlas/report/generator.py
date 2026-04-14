"""Report generator — compile research results into structured reports.

Takes the agent's answer, evaluation results, and sources,
then produces a structured markdown report with:
- Executive summary
- Main findings with inline citations
- Confidence indicators per section
- Grounding analysis
- Bibliography
"""

from __future__ import annotations

from datetime import UTC, datetime

from atlas.evaluator.confidence import EvaluationReport
from atlas.observability.logger import get_logger
from atlas.report.citations import CitationManager

log = get_logger(__name__)


class ReportGenerator:
    """Generate structured research reports from agent results."""

    def generate(
        self,
        query: str,
        answer: str,
        sources: list[dict],
        evaluation: EvaluationReport | None = None,
        dag_summary: dict | None = None,
    ) -> str:
        """Generate a full markdown research report.

        Args:
            query: The original research question.
            answer: The agent's synthesized answer.
            sources: List of source dicts from the agent.
            evaluation: Optional evaluation report with grounding scores.
            dag_summary: Optional DAG execution summary.

        Returns:
            Formatted markdown report string.
        """
        citation_mgr = CitationManager()
        sections = []

        # --- Header ---
        sections.append(self._build_header(query))

        # --- Confidence Banner ---
        if evaluation and evaluation.total_claims > 0:
            sections.append(self._build_confidence_banner(evaluation))

        # --- Main Findings ---
        sections.append(self._build_findings(answer, sources, citation_mgr))

        # --- Grounding Analysis ---
        if evaluation and evaluation.total_claims > 0:
            sections.append(self._build_grounding_analysis(evaluation))

        # --- Research Process ---
        if dag_summary:
            sections.append(self._build_process_summary(dag_summary))

        # --- Sources & Bibliography ---
        self._register_sources(sources, citation_mgr)
        sections.append(citation_mgr.format_bibliography())

        # --- Footer ---
        sections.append(self._build_footer())

        report = "\n\n".join(s for s in sections if s)

        log.info(
            "report_generated",
            query=query[:60],
            length=len(report),
            num_citations=citation_mgr.num_citations,
        )

        return report

    def _build_header(self, query: str) -> str:
        """Build the report header."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"# ATLAS Research Report\n\n"
            f"**Research Question:** {query}\n\n"
            f"**Generated:** {timestamp}\n\n"
            f"---"
        )

    def _build_confidence_banner(self, evaluation: EvaluationReport) -> str:
        """Build confidence indicator banner."""
        confidence = evaluation.overall_confidence
        pct = round(confidence * 100)

        if confidence >= 0.8:
            indicator = "HIGH"
            emoji = "🟢"
        elif confidence >= 0.5:
            indicator = "MODERATE"
            emoji = "🟡"
        else:
            indicator = "LOW"
            emoji = "🔴"

        return (
            f"## Confidence: {emoji} {indicator} ({pct}%)\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Total Claims Analyzed | {evaluation.total_claims} |\n"
            f"| Supported by Evidence | {evaluation.supported_claims} |\n"
            f"| Unsupported | {evaluation.unsupported_claims} |\n"
            f"| Contradicted | {evaluation.contradicted_claims} |"
        )

    def _build_findings(
        self,
        answer: str,
        sources: list[dict],
        citation_mgr: CitationManager,
    ) -> str:
        """Build the main findings section."""
        return f"## Findings\n\n{answer}"

    def _build_grounding_analysis(
        self, evaluation: EvaluationReport
    ) -> str:
        """Build detailed grounding analysis per claim."""
        lines = ["## Grounding Analysis", ""]

        for i, claim_result in enumerate(evaluation.claim_results, 1):
            status = claim_result["status"]
            score = claim_result["score"]

            if status == "supported":
                icon = "✅"
            elif status == "contradicted":
                icon = "❌"
            else:
                icon = "⚠️"

            source = claim_result.get("evidence_source", "unknown")

            lines.append(
                f"{i}. {icon} **{status.upper()}** "
                f"(score: {score:.2f}) — "
                f"*{claim_result['claim']}*"
            )
            lines.append(f"   Source: {source}")
            lines.append("")

        return "\n".join(lines)

    def _build_process_summary(self, dag_summary: dict) -> str:
        """Build research process summary from DAG."""
        lines = [
            "## Research Process",
            "",
            f"The research was decomposed into "
            f"**{dag_summary.get('num_tasks', 0)} sub-tasks**.",
            "",
        ]

        tasks = dag_summary.get("tasks", [])
        for task in tasks:
            status = task.get("status", "unknown")
            icon = "✅" if status == "completed" else "❌"
            lines.append(
                f"- {icon} **{task.get('query', 'Unknown task')}**"
            )

        return "\n".join(lines)

    def _register_sources(
        self, sources: list[dict], citation_mgr: CitationManager
    ) -> None:
        """Register all sources as citations."""
        seen = set()
        for source in sources:
            tool = source.get("tool", "unknown")
            query = source.get("query", "")
            key = f"{tool}:{query}"

            if key in seen:
                continue
            seen.add(key)

            citation_mgr.add_source(
                source=f"{tool}: {query}",
                title=query,
                content_preview=source.get("result_preview", ""),
            )

    def _build_footer(self) -> str:
        """Build the report footer."""
        return (
            "---\n\n"
            "*Generated by ATLAS — "
            "Autonomous Tool-using LLM Agent for Synthesis*"
        )
