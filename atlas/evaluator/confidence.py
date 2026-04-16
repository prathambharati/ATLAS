"""Confidence scoring — aggregate claim-level grounding into overall scores."""

from __future__ import annotations

from dataclasses import dataclass, field

from atlas.evaluator.grounding import GroundingResult
from atlas.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Full evaluation report for a generated response."""

    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    overall_confidence: float = 0.0
    claim_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "contradicted_claims": self.contradicted_claims,
            "overall_confidence": round(self.overall_confidence, 4),
            "claim_results": self.claim_results,
        }


class ConfidenceScorer:
    """Aggregate claim-level grounding scores into overall confidence."""

    def __init__(self, support_threshold: float = 0.5):
        self.support_threshold = support_threshold

    def evaluate(
        self,
        grounding_results: list[GroundingResult],
    ) -> EvaluationReport:
        """Produce an evaluation report from grounding results."""
        if not grounding_results:
            return EvaluationReport()

        supported = 0
        unsupported = 0
        contradicted = 0
        claim_details = []

        for result in grounding_results:
            # Check contradicted FIRST — a contradiction should never
            # be counted as supported even if score is high
            if result.is_contradicted:
                contradicted += 1
                status = "contradicted"
            elif result.is_supported:
                supported += 1
                status = "supported"
            else:
                unsupported += 1
                status = "unsupported"

            claim_details.append({
                "claim": result.claim,
                "status": status,
                "score": round(result.score, 4),
                "label": result.label,
                "evidence_source": result.evidence_source,
                "evidence_preview": result.evidence[:150],
            })

        total = len(grounding_results)
        confidence = supported / total if total > 0 else 0.0

        report = EvaluationReport(
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=unsupported,
            contradicted_claims=contradicted,
            overall_confidence=confidence,
            claim_results=claim_details,
        )

        log.info(
            "evaluation_complete",
            total=total,
            supported=supported,
            unsupported=unsupported,
            contradicted=contradicted,
            confidence=round(confidence, 4),
        )

        return report
