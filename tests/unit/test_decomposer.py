"""Tests for the query decomposer.

Unit tests (no API needed) test the DAG building logic.
Integration tests (need OPENAI_API_KEY) test the full LLM decomposition.
"""

import pytest

from atlas.planner.decomposer import QueryDecomposer


class TestDecomposerBuildDAG:
    """Test the _build_dag method without needing an API call."""

    def setup_method(self):
        """Create decomposer but we won't call the LLM in these tests."""
        self.decomposer = QueryDecomposer.__new__(QueryDecomposer)

    def test_build_dag_simple_plan(self):
        """Should build a DAG from a valid plan."""
        plan = {
            "tasks": [
                {"id": "t1", "query": "What is X?", "description": "Define X", "depends_on": []},
                {"id": "t2", "query": "What is Y?", "description": "Define Y", "depends_on": []},
                {
                    "id": "t3",
                    "query": "Compare X and Y",
                    "description": "Synthesis",
                    "depends_on": ["t1", "t2"],
                },
            ]
        }

        dag = self.decomposer._build_dag(plan)
        assert dag.num_tasks == 3
        assert dag.get_task("t3").depends_on == ["t1", "t2"]

    def test_build_dag_linear_chain(self):
        """Should handle a linear chain of dependencies."""
        plan = {
            "tasks": [
                {"id": "t1", "query": "Step 1", "description": "", "depends_on": []},
                {"id": "t2", "query": "Step 2", "description": "", "depends_on": ["t1"]},
                {"id": "t3", "query": "Step 3", "description": "", "depends_on": ["t2"]},
            ]
        }

        dag = self.decomposer._build_dag(plan)
        batches = dag.execution_order()
        assert len(batches) == 3

    def test_build_dag_empty_tasks_raises(self):
        """Should raise ValueError for empty task list."""
        with pytest.raises(ValueError, match="empty"):
            self.decomposer._build_dag({"tasks": []})

    def test_build_dag_filters_invalid_deps(self):
        """Should silently filter out dependencies that don't exist."""
        plan = {
            "tasks": [
                {"id": "t1", "query": "Task 1", "description": "", "depends_on": []},
                {
                    "id": "t2",
                    "query": "Task 2",
                    "description": "",
                    "depends_on": ["t1", "nonexistent"],
                },
            ]
        }

        dag = self.decomposer._build_dag(plan)
        assert dag.num_tasks == 2
        # Should only have t1 as dependency, nonexistent filtered out
        task2 = dag.get_task("t2")
        assert "nonexistent" not in task2.depends_on

    def test_build_dag_single_task(self):
        """Should handle a plan with just one task."""
        plan = {
            "tasks": [
                {
                    "id": "t1",
                    "query": "Simple question",
                    "description": "Answer it",
                    "depends_on": [],
                },
            ]
        }

        dag = self.decomposer._build_dag(plan)
        assert dag.num_tasks == 1
        batches = dag.execution_order()
        assert len(batches) == 1


class TestDecomposerFallback:
    """Test fallback behavior when decomposition fails."""

    def setup_method(self):
        self.decomposer = QueryDecomposer.__new__(QueryDecomposer)

    def test_fallback_dag(self):
        """Fallback should create a single-task DAG."""
        dag = self.decomposer._fallback_dag("Some complex query")
        assert dag.num_tasks == 1
        task = dag.get_task("t1")
        assert task.query == "Some complex query"
        assert "Fallback" in task.description


class TestDecomposerIntegration:
    """Integration tests that hit the OpenAI API.

    These are skipped if OPENAI_API_KEY is not set.
    Run with: pytest tests/unit/test_decomposer.py -v -k integration
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from atlas.config import settings

        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY not set")
        self.decomposer = QueryDecomposer()

    def test_classify_simple_query(self):
        """Simple factual query should be classified as simple."""
        result = self.decomposer.is_complex("What is BERT?")
        assert result is False

    def test_classify_complex_query(self):
        """Comparative query should be classified as complex."""
        result = self.decomposer.is_complex(
            "Compare speculative decoding vs continuous batching"
            " for LLM inference optimization"
        )
        assert result is True

    def test_decompose_complex_query(self):
        """Should decompose a complex query into multiple sub-tasks."""
        dag = self.decomposer.decompose(
            "What are the key differences between BERT and GPT"
            " architectures, and which is better for classification?"
        )

        assert dag.num_tasks >= 2
        assert dag.num_tasks <= 6

        # Should have at least one synthesis task that depends on others
        has_dependency = any(t.depends_on for t in dag.tasks)
        assert has_dependency, "Complex query should have tasks with dependencies"

        # Execution order should work without errors
        batches = dag.execution_order()
        assert len(batches) >= 2

    def test_decompose_simple_query(self):
        """Simple query should get a single-task DAG."""
        dag = self.decomposer.decompose("What is attention in transformers?")
        assert dag.num_tasks == 1
