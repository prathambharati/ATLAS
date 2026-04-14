"""Evaluator module — hallucination detection via NLI + confidence scoring."""

from atlas.evaluator.confidence import ConfidenceScorer, EvaluationReport
from atlas.evaluator.evaluator import HallucinationEvaluator
from atlas.evaluator.grounding import GroundingResult, GroundingScorer

__all__ = [
    "HallucinationEvaluator",
    "ConfidenceScorer",
    "EvaluationReport",
    "GroundingScorer",
    "GroundingResult",
]
