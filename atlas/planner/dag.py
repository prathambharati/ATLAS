"""Directed Acyclic Graph (DAG) for task planning.

Represents a research query decomposed into sub-tasks with dependencies.
Supports topological sorting for correct execution order and parallel
execution of independent tasks.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from atlas.observability.logger import get_logger

log = get_logger(__name__)


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """A single sub-task in the research plan."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "query": self.query,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


class TaskDAG:
    """Directed Acyclic Graph of research sub-tasks.

    Manages task dependencies and provides topological ordering
    for correct execution. Tasks with no dependencies can run
    in parallel (returned together in the same batch).

    Example:
        dag = TaskDAG()
        t1 = dag.add_task("What is speculative decoding?")
        t2 = dag.add_task("What is continuous batching?")
        t3 = dag.add_task("Compare the two approaches", depends_on=[t1, t2])

        for batch in dag.execution_order():
            # batch 1: [t1, t2] (parallel)
            # batch 2: [t3] (depends on both)
            for task in batch:
                execute(task)
    """

    def __init__(self):
        self._tasks: dict[str, Task] = {}

    def add_task(
        self,
        query: str,
        description: str = "",
        depends_on: list[str] | None = None,
        task_id: str | None = None,
    ) -> str:
        """Add a task to the DAG.

        Args:
            query: The sub-question to answer.
            description: What this task does.
            depends_on: List of task IDs this task depends on.
            task_id: Optional custom task ID.

        Returns:
            The task ID.

        Raises:
            ValueError: If a dependency doesn't exist or would create a cycle.
        """
        depends_on = depends_on or []

        # Validate dependencies exist
        for dep_id in depends_on:
            if dep_id not in self._tasks:
                raise ValueError(f"Dependency '{dep_id}' does not exist in the DAG")

        task = Task(
            task_id=task_id or str(uuid.uuid4())[:8],
            query=query,
            description=description,
            depends_on=depends_on,
        )

        # Check for cycles before adding
        self._tasks[task.task_id] = task
        if self._has_cycle():
            del self._tasks[task.task_id]
            raise ValueError(f"Adding task '{task.task_id}' would create a cycle")

        log.info(
            "task_added",
            task_id=task.task_id,
            query=query[:60],
            depends_on=depends_on,
        )
        return task.task_id

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def mark_completed(self, task_id: str, result: str) -> None:
        """Mark a task as completed with its result."""
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.result = result

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed."""
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.error = error

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks whose dependencies are all completed.

        Returns:
            List of tasks that are ready to execute.
        """
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            deps_met = all(
                self._tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.depends_on
                if dep_id in self._tasks
            )

            if deps_met:
                ready.append(task)

        return ready

    def execution_order(self) -> list[list[Task]]:
        """Get the topological execution order as batches.

        Each batch contains tasks that can run in parallel.
        Batches must be executed sequentially.

        Returns:
            List of batches, where each batch is a list of Tasks.

        Raises:
            ValueError: If the DAG has a cycle.
        """
        if self._has_cycle():
            raise ValueError("DAG contains a cycle — cannot determine execution order")

        # Kahn's algorithm for topological sort with batching
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        for task in self._tasks.values():
            for dep_id in task.depends_on:
                if dep_id in in_degree:
                    in_degree[task.task_id] += 1

        # Start with tasks that have no dependencies
        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        batches: list[list[Task]] = []

        while queue:
            # All tasks currently in queue can run in parallel
            batch = []
            next_queue: list[str] = []

            for _ in range(len(queue)):
                tid = queue.popleft()
                batch.append(self._tasks[tid])

                # Reduce in-degree for dependents
                for other_task in self._tasks.values():
                    if tid in other_task.depends_on:
                        in_degree[other_task.task_id] -= 1
                        if in_degree[other_task.task_id] == 0:
                            next_queue.append(other_task.task_id)

            batches.append(batch)
            queue.extend(next_queue)

        return batches

    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(t.status == TaskStatus.COMPLETED for t in self._tasks.values())

    def get_results(self) -> dict[str, str]:
        """Get results from all completed tasks."""
        return {tid: task.result for tid, task in self._tasks.items() if task.result is not None}

    def _has_cycle(self) -> bool:
        """Detect cycles using DFS."""
        white, gray, black = 0, 1, 2
        color = {tid: white for tid in self._tasks}

        def dfs(tid: str) -> bool:
            color[tid] = gray
            task = self._tasks[tid]
            for dep_id in task.depends_on:
                if dep_id not in color:
                    continue
                if color[dep_id] == gray:
                    return True  # Back edge = cycle
                if color[dep_id] == white and dfs(dep_id):
                    return True
            color[tid] = black
            return False

        return any(color[tid] == white and dfs(tid) for tid in self._tasks)

    @property
    def num_tasks(self) -> int:
        return len(self._tasks)

    @property
    def tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def summary(self) -> dict:
        """Get a summary of the DAG state."""
        return {
            "num_tasks": self.num_tasks,
            "completed": sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED),
            "pending": sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING),
            "failed": sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED),
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }

    def __repr__(self) -> str:
        return f"TaskDAG(tasks={self.num_tasks})"
