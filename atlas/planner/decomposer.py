"""Query decomposer using LLM to break complex questions into a task DAG.

Takes a research question, uses an LLM to decompose it into sub-questions
with dependencies, and returns a TaskDAG ready for execution.

Simple queries (single-concept) skip decomposition and go straight
to retrieval. Complex queries get broken into a DAG.
"""

import json
import time

from openai import OpenAI

from atlas.config import settings
from atlas.observability.logger import get_logger
from atlas.observability.tracer import StepType, TraceStep
from atlas.planner.dag import TaskDAG
from atlas.planner.prompts import (
    DECOMPOSER_SYSTEM_PROMPT,
    DECOMPOSER_USER_PROMPT,
    SIMPLE_QUERY_SYSTEM_PROMPT,
)

log = get_logger(__name__)


class QueryDecomposer:
    """Decompose complex research queries into executable task DAGs."""

    def __init__(self, model: str | None = None):
        """
        Args:
            model: OpenAI model to use. Defaults to settings.llm_model.
        """
        self.model = model or settings.llm_model
        self._client = OpenAI(api_key=settings.openai_api_key)
        log.info("decomposer_ready", model=self.model)

    def is_complex(self, query: str) -> bool:
        """Classify whether a query needs decomposition.

        Simple queries are answered directly. Complex queries
        get decomposed into a DAG of sub-tasks.

        Args:
            query: The research question.

        Returns:
            True if the query is complex and needs decomposition.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SIMPLE_QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=10,
            )
            answer = response.choices[0].message.content.strip().lower()
            is_complex = answer == "complex"

            log.info("query_classified", query=query[:60], classification=answer)
            return is_complex

        except Exception as e:
            log.error("classification_failed", error=str(e))
            # Default to complex (safer — will decompose)
            return True

    def decompose(self, query: str) -> TaskDAG:
        """Decompose a query into a TaskDAG.

        If the query is simple, creates a single-task DAG.
        If complex, uses the LLM to break it into sub-tasks.

        Args:
            query: The research question.

        Returns:
            A TaskDAG with sub-tasks and dependencies.
        """
        # Check complexity first
        if not self.is_complex(query):
            log.info("simple_query_no_decomposition", query=query[:60])
            dag = TaskDAG()
            dag.add_task(
                query=query,
                description="Direct answer — simple query",
                task_id="t1",
            )
            return dag

        # Complex query — decompose with LLM
        return self._decompose_with_llm(query)

    def _decompose_with_llm(self, query: str) -> TaskDAG:
        """Use the LLM to decompose a complex query into sub-tasks.

        Args:
            query: The complex research question.

        Returns:
            A TaskDAG built from the LLM's decomposition.
        """
        start_time = time.time()

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": DECOMPOSER_USER_PROMPT.format(query=query),
                    },
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = response.usage.completion_tokens if response.usage else 0

            # Parse LLM response
            plan = json.loads(raw)
            dag = self._build_dag(plan)

            end_time = time.time()

            log.info(
                "decomposition_complete",
                query=query[:60],
                num_tasks=dag.num_tasks,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=round((end_time - start_time) * 1000, 2),
            )

            return dag

        except json.JSONDecodeError as e:
            log.error("decomposition_json_error", error=str(e), raw=raw[:200])
            # Fallback: single-task DAG
            return self._fallback_dag(query)

        except Exception as e:
            log.error("decomposition_failed", error=str(e))
            return self._fallback_dag(query)

    def _build_dag(self, plan: dict) -> TaskDAG:
        """Build a TaskDAG from the LLM's JSON plan.

        Args:
            plan: Parsed JSON with "tasks" key.

        Returns:
            A validated TaskDAG.
        """
        dag = TaskDAG()
        tasks = plan.get("tasks", [])

        if not tasks:
            raise ValueError("LLM returned empty task list")

        # First pass: add all tasks without dependencies
        # (so dependency references can be validated)
        task_ids = set()
        for task_data in tasks:
            tid = task_data.get("id", str(len(task_ids)))
            task_ids.add(tid)

        # Second pass: add with dependencies
        for task_data in tasks:
            tid = task_data.get("id")
            query = task_data.get("query", "")
            description = task_data.get("description", "")
            depends_on = task_data.get("depends_on", [])

            # Filter out invalid dependencies
            valid_deps = [d for d in depends_on if d in task_ids and d != tid]

            # Add tasks that this depends on first (if not already added)
            dag.add_task(
                query=query,
                description=description,
                depends_on=[d for d in valid_deps if dag.get_task(d) is not None],
                task_id=tid,
            )

        return dag

    def _fallback_dag(self, query: str) -> TaskDAG:
        """Create a simple single-task DAG as fallback."""
        log.warning("using_fallback_dag", query=query[:60])
        dag = TaskDAG()
        dag.add_task(
            query=query,
            description="Fallback — decomposition failed, answering directly",
            task_id="t1",
        )
        return dag

    def create_trace_step(
        self,
        query: str,
        dag: TaskDAG,
        start_time: float,
        end_time: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> TraceStep:
        """Create a trace step for the decomposition."""
        return TraceStep(
            step_id=f"decompose_{dag.num_tasks}",
            step_type=StepType.LLM_CALL,
            input_data={"query": query},
            output_data=dag.summary(),
            start_time=start_time,
            end_time=end_time,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
