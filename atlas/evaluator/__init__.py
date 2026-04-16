"""Evaluator module — hallucination detection + self-correction."""

from atlas.evaluator.confidence import ConfidenceScorer, EvaluationReport
from atlas.evaluator.evaluator import HallucinationEvaluator
from atlas.evaluator.grounding import GroundingResult, GroundingScorer
from atlas.evaluator.self_corrector import SelfCorrector

__all__ = [
    "HallucinationEvaluator",
    "ConfidenceScorer",
    "EvaluationReport",
    "GroundingScorer",
    "GroundingResult",
    "SelfCorrector",
]
