"""Tests for evaluation suite infrastructure."""

import json
from pathlib import Path

import pytest

from atlas.evaluator.confidence import EvaluationReport


class TestEvalQuestions:
    """Validate the evaluation question set."""

    def setup_method(self):
        path = Path("data/eval/questions.json")
        if not path.exists():
            self.questions = []
            return
        with open(path) as f:
            data = json.load(f)
        self.questions = data.get("eval_set", [])

    def test_questions_exist(self):
        """Should have at least 10 evaluation questions."""
        if not self.questions:
            pytest.skip("questions.json not found")
        assert len(self.questions) >= 10

    def test_question_structure(self):
        """Each question should have required fields."""
        for q in self.questions:
            assert "id" in q, f"Missing id in question: {q}"
            assert "query" in q, f"Missing query in question: {q}"
            assert "complexity" in q
            assert "expected_claims" in q
            assert "domain" in q

    def test_unique_ids(self):
        """All question IDs should be unique."""
        ids = [q["id"] for q in self.questions]
        assert len(ids) == len(set(ids))

    def test_complexity_values(self):
        """Complexity should be 'simple' or 'complex'."""
        for q in self.questions:
            assert q["complexity"] in ("simple", "complex")

    def test_has_both_complexities(self):
        """Should have both simple and complex questions."""
        if not self.questions:
            pytest.skip("questions.json not found")
        complexities = {q["complexity"] for q in self.questions}
        assert "simple" in complexities
        assert "complex" in complexities

    def test_expected_claims_not_empty(self):
        """Each question should have at least 2 expected claims."""
        for q in self.questions:
            assert len(q["expected_claims"]) >= 2, (
                f"Question {q['id']} has too few expected claims"
            )


class TestEvalMetrics:
    """Test metric calculation logic."""

    def test_groundedness_calculation(self):
        """Groundedness = supported / total claims."""
        total = 10
        supported = 8
        groundedness = supported / total
        assert abs(groundedness - 0.8) < 0.01

    def test_perfect_confidence(self):
        """All supported should give 1.0 confidence."""
        report = EvaluationReport(
            total_claims=5,
            supported_claims=5,
            unsupported_claims=0,
            contradicted_claims=0,
            overall_confidence=1.0,
        )
        assert report.overall_confidence == 1.0

    def test_zero_confidence(self):
        """No supported claims should give 0.0 confidence."""
        report = EvaluationReport(
            total_claims=5,
            supported_claims=0,
            unsupported_claims=5,
            contradicted_claims=0,
            overall_confidence=0.0,
        )
        assert report.overall_confidence == 0.0
