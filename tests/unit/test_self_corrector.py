"""Tests for self-corrector and cost tracker."""

import pytest

from atlas.observability.metrics import CostTracker


class TestCostTracker:
    """Test cost tracking."""

    def test_record_and_summary(self):
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record(
            query="test query",
            tokens_in=1000,
            tokens_out=500,
            trace_id="abc123",
        )

        summary = tracker.get_summary()
        assert summary["total_queries"] == 1
        assert summary["total_tokens"] == 1500
        assert summary["total_cost_usd"] > 0

    def test_multiple_queries(self):
        tracker = CostTracker(model="gpt-4o-mini")
        tracker.record("q1", tokens_in=100, tokens_out=50)
        tracker.record("q2", tokens_in=200, tokens_out=100)
        tracker.record("q3", tokens_in=300, tokens_out=150)

        summary = tracker.get_summary()
        assert summary["total_queries"] == 3
        assert summary["total_tokens"] == 900
        assert summary["avg_tokens_per_query"] == 300

    def test_empty_tracker(self):
        tracker = CostTracker()
        summary = tracker.get_summary()
        assert summary["total_queries"] == 0
        assert summary["total_cost_usd"] == 0.0

    def test_cost_calculation(self):
        """gpt-4o-mini: $0.15/1M input, $0.60/1M output."""
        tracker = CostTracker(model="gpt-4o-mini")
        entry = tracker.record("test", tokens_in=1_000_000, tokens_out=0)
        assert abs(entry.cost_usd - 0.15) < 0.001

        entry2 = tracker.record("test2", tokens_in=0, tokens_out=1_000_000)
        assert abs(entry2.cost_usd - 0.60) < 0.001

    def test_get_recent(self):
        tracker = CostTracker()
        for i in range(15):
            tracker.record(f"query_{i}", tokens_in=100, tokens_out=50)

        recent = tracker.get_recent(n=5)
        assert len(recent) == 5
        assert recent[-1]["query"].startswith("query_14")


class TestSelfCorrectorUnit:
    """Unit tests for self-corrector (no API calls)."""

    def test_correction_cycle_to_dict(self):
        from atlas.evaluator.self_corrector import CorrectionCycle

        cycle = CorrectionCycle(
            cycle_number=1,
            claims_before=10,
            supported_before=6,
            unsupported_before=4,
            confidence_before=0.6,
            requery_count=3,
            new_evidence_count=5,
            claims_after=10,
            supported_after=8,
            confidence_after=0.8,
            latency_ms=5000.0,
        )

        d = cycle.to_dict()
        assert d["cycle"] == 1
        assert d["before"]["confidence"] == 0.6
        assert d["after"]["confidence"] == 0.8
        assert d["improvement"] == 0.2

    def test_correction_result_to_dict(self):
        from atlas.evaluator.self_corrector import (
            SelfCorrectionResult,
        )

        result = SelfCorrectionResult(
            original_answer="Original text",
            corrected_answer="Corrected text",
            original_confidence=0.5,
            final_confidence=0.85,
            num_cycles=2,
            converged=True,
        )

        d = result.to_dict()
        assert d["original_confidence"] == 0.5
        assert d["final_confidence"] == 0.85
        assert d["improvement"] == 0.35
        assert d["converged"] is True


class TestSelfCorrectorIntegration:
    """Integration tests (require API keys)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from atlas.config import settings

        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY not set")

    def test_self_correction_improves_confidence(self):
        from atlas.evaluator.self_corrector import SelfCorrector

        corrector = SelfCorrector(max_cycles=1)

        answer = (
            "BERT uses bidirectional attention to process text. "
            "It was developed by OpenAI in 2017. "
            "BERT has 340 million parameters in its large variant."
        )

        evidence = [
            {
                "text": (
                    "BERT is a bidirectional language model introduced "
                    "by Google AI Language researchers in 2018. "
                    "BERT-Large has 340 million parameters."
                ),
                "source": "bert_paper.pdf",
            },
        ]

        result = corrector.correct(answer, evidence)

        assert result.corrected_answer != result.original_answer
        assert result.num_cycles >= 1
        assert isinstance(result.final_confidence, float)
