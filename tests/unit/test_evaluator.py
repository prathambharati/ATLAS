"""Tests for the hallucination detection pipeline."""

import pytest

from atlas.evaluator.confidence import ConfidenceScorer
from atlas.evaluator.grounding import GroundingResult


class TestGroundingResult:
    """Test the GroundingResult dataclass."""

    def test_supported_claim(self):
        result = GroundingResult(
            claim="BERT uses bidirectional attention",
            label="entailment",
            score=0.92,
            evidence="BERT employs bidirectional self-attention.",
            evidence_source="paper.pdf",
        )
        assert result.is_supported is True
        assert result.is_contradicted is False

    def test_contradicted_claim(self):
        result = GroundingResult(
            claim="GPT is bidirectional",
            label="contradiction",
            score=0.85,
            evidence="GPT uses unidirectional (left-to-right) attention.",
            evidence_source="paper.pdf",
        )
        assert result.is_supported is False
        assert result.is_contradicted is True

    def test_neutral_claim(self):
        result = GroundingResult(
            claim="The model has 175B parameters",
            label="neutral",
            score=0.6,
            evidence="The model was trained on a large corpus.",
            evidence_source="paper.pdf",
        )
        assert result.is_supported is False
        assert result.is_contradicted is False

    def test_low_score_not_supported(self):
        """Even entailment label needs score >= 0.5."""
        result = GroundingResult(
            claim="Some claim",
            label="entailment",
            score=0.3,
            evidence="Some evidence",
            evidence_source="test",
        )
        assert result.is_supported is False

    def test_to_dict(self):
        result = GroundingResult(
            claim="Test claim",
            label="entailment",
            score=0.9,
            evidence="Test evidence text here",
            evidence_source="source.pdf",
        )
        d = result.to_dict()
        assert d["claim"] == "Test claim"
        assert d["label"] == "entailment"
        assert d["is_supported"] is True
        assert "evidence_preview" in d


class TestConfidenceScorer:
    """Test confidence score aggregation."""

    def test_all_supported(self):
        results = [
            GroundingResult("c1", "entailment", 0.9, "e1", "s1"),
            GroundingResult("c2", "entailment", 0.8, "e2", "s2"),
            GroundingResult("c3", "entailment", 0.7, "e3", "s3"),
        ]
        scorer = ConfidenceScorer()
        report = scorer.evaluate(results)

        assert report.total_claims == 3
        assert report.supported_claims == 3
        assert report.overall_confidence == 1.0

    def test_mixed_results(self):
        results = [
            GroundingResult("c1", "entailment", 0.9, "e1", "s1"),
            GroundingResult("c2", "neutral", 0.6, "e2", "s2"),
            GroundingResult("c3", "contradiction", 0.8, "e3", "s3"),
        ]
        scorer = ConfidenceScorer()
        report = scorer.evaluate(results)

        assert report.total_claims == 3
        assert report.supported_claims == 1
        assert report.unsupported_claims == 1
        assert report.contradicted_claims == 1
        assert abs(report.overall_confidence - 1 / 3) < 0.01

    def test_empty_results(self):
        scorer = ConfidenceScorer()
        report = scorer.evaluate([])
        assert report.total_claims == 0
        assert report.overall_confidence == 0.0

    def test_report_to_dict(self):
        results = [
            GroundingResult("c1", "entailment", 0.9, "e1", "s1"),
        ]
        scorer = ConfidenceScorer()
        report = scorer.evaluate(results)
        d = report.to_dict()

        assert "total_claims" in d
        assert "overall_confidence" in d
        assert "claim_results" in d
        assert len(d["claim_results"]) == 1


class TestGroundingScorer:
    """Test NLI-based grounding scoring.

    These tests load the BART-MNLI model (~1.6GB first time).
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from atlas.evaluator.grounding import GroundingScorer

        self.scorer = GroundingScorer()

    def test_supported_claim(self):
        """Claim matching evidence should be entailed."""
        result = self.scorer.score_claim(
            claim="Transformers use self-attention mechanisms",
            evidence_chunks=[
                {
                    "text": (
                        "The Transformer architecture relies entirely "
                        "on self-attention to compute representations."
                    ),
                    "source": "paper.pdf",
                },
            ],
        )
        assert result.label == "entailment"
        assert result.score > 0.5

    def test_unsupported_claim(self):
        """Supported claim should be entailment, unrelated should not."""
        supported = self.scorer.score_claim(
            claim="Transformers use self-attention",
            evidence_chunks=[
                {
                    "text": "The Transformer relies on self-attention mechanisms.",
                    "source": "paper.pdf",
                },
            ],
        )
        unrelated = self.scorer.score_claim(
            claim="The model has 175 billion parameters",
            evidence_chunks=[
                {
                    "text": "The Transformer relies on self-attention mechanisms.",
                    "source": "paper.pdf",
                },
            ],
        )
        assert supported.label == "entailment"
        assert unrelated.label == "neutral"

    def test_no_evidence(self):
        """No evidence should return neutral with 0 score."""
        result = self.scorer.score_claim(
            claim="Any claim here",
            evidence_chunks=[],
        )
        assert result.label == "neutral"
        assert result.score == 0.0

    def test_multiple_chunks_returns_result(self):
        """Should process multiple chunks and return a valid result."""
        result = self.scorer.score_claim(
            claim="BERT is a bidirectional model",
            evidence_chunks=[
                {
                    "text": "The weather is sunny today.",
                    "source": "weather.txt",
                },
                {
                    "text": (
                        "BERT uses bidirectional self-attention "
                        "to process text in both directions."
                    ),
                    "source": "bert_paper.pdf",
                },
            ],
        )
        assert result.label in ("entailment", "contradiction", "neutral")
        assert result.score > 0
        assert result.evidence_source in ("weather.txt", "bert_paper.pdf")


class TestHallucinationEvaluatorIntegration:
    """Full pipeline integration test.

    Requires OPENAI_API_KEY for claim extraction.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from atlas.config import settings

        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY not set")

    def test_full_pipeline(self):
        from atlas.evaluator.evaluator import HallucinationEvaluator

        evaluator = HallucinationEvaluator()

        generated = (
            "BERT uses bidirectional self-attention to process text. "
            "It was developed by Google in 2018. "
            "BERT has 175 billion parameters."
        )

        evidence = [
            {
                "text": (
                    "BERT employs bidirectional self-attention, "
                    "processing text in both directions simultaneously. "
                    "It was introduced by Google researchers in 2018."
                ),
                "source": "bert_paper.pdf",
            },
        ]

        report = evaluator.evaluate(generated, evidence)

        assert report.total_claims >= 2
        assert report.supported_claims >= 1
        # "175 billion parameters" should be unsupported
        assert report.overall_confidence > 0
        assert report.overall_confidence < 1.0
        assert len(report.claim_results) == report.total_claims
