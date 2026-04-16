"""ReAct Agent Orchestrator — the core brain of ATLAS.

Takes a research query, uses the planner to decompose it into sub-tasks,
then executes each task using a ReAct loop (Reason → Act → Observe).
The agent calls tools (retrieval, web search, arxiv, calculator) to
gather evidence, then synthesizes a final answer.
"""

import json
import time

from openai import OpenAI

from atlas.agent.memory import AgentMemory
from atlas.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT,
    TASK_EXECUTION_PROMPT,
)
from atlas.config import settings
from atlas.observability.logger import get_logger
from atlas.observability.tracer import StepType, Trace, TraceStep, trace_store
from atlas.planner.decomposer import QueryDecomposer
from atlas.tools.default_tools import build_default_tools
from atlas.tools.registry import ToolRegistry

log = get_logger(__name__)


class AgentOrchestrator:
    """ReAct-style agent that executes research tasks with tool use.

    Flow:
        1. Planner decomposes query into a TaskDAG
        2. For each batch of tasks (topological order):
           a. Build context from prior task results
           b. Run ReAct loop: LLM reasons, calls tools, observes results
           c. Store task result in memory
        3. Synthesize all task results into final answer
    """

    def __init__(
        self,
        model: str | None = None,
        max_steps: int = 10,
        tool_registry: ToolRegistry | None = None,
    ):
        """
        Args:
            model: OpenAI model to use for the agent.
            max_steps: Maximum tool-calling steps per task.
            tool_registry: Pre-built tool registry. If None, builds defaults.
        """
        self.model = model or settings.llm_model
        self.max_steps = max_steps
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._decomposer = QueryDecomposer(model=self.model)
        self._tools = tool_registry or build_default_tools()

        log.info(
            "agent_ready",
            model=self.model,
            max_steps=max_steps,
            tools=self._tools.tool_names,
        )

    def run(self, query: str) -> dict:
        """Run the full research pipeline for a query.

        Args:
            query: The research question.

        Returns:
            Dict with keys: trace_id, query, answer, sources, dag_summary
        """
        trace = trace_store.create(query=query)
        memory = AgentMemory()
        start_time = time.time()

        log.info("research_started", query=query[:80], trace_id=trace.trace_id)

        # Step 1: Decompose query into task DAG
        dag = self._decomposer.decompose(query)
        log.info(
            "plan_created",
            num_tasks=dag.num_tasks,
            trace_id=trace.trace_id,
        )

        # Step 2: Execute tasks in topological order
        batches = dag.execution_order()
        for batch_idx, batch in enumerate(batches):
            log.info(
                "executing_batch",
                batch=batch_idx + 1,
                total_batches=len(batches),
                tasks=[t.task_id for t in batch],
            )

            for task in batch:
                task_result = self._execute_task(
                    task_query=task.query,
                    task_description=task.description,
                    memory=memory,
                    trace=trace,
                )
                dag.mark_completed(task.task_id, task_result)
                memory.add_task_result(task.task_id, task_result)

        # Step 3: Synthesize final answer
        if dag.num_tasks > 1:
            answer = self._synthesize(query, memory, trace)
        else:
            # Single task — its result is the answer
            results = dag.get_results()
            answer = list(results.values())[0] if results else "No answer generated."

        end_time = time.time()

        log.info(
            "research_complete",
            trace_id=trace.trace_id,
            total_time_s=round(end_time - start_time, 2),
            total_steps=len(trace.steps),
            total_tokens=trace.total_tokens,
        )

        return {
            "trace_id": trace.trace_id,
            "query": query,
            "answer": answer,
            "sources": memory.sources,
            "dag_summary": dag.summary(),
        }

    def _execute_task(
        self,
        task_query: str,
        task_description: str,
        memory: AgentMemory,
        trace: Trace,
    ) -> str:
        """Execute a single task using the ReAct loop.

        The agent reasons about what tool to call, calls it,
        observes the result, and repeats until it has an answer.
        """
        # Build context from prior task results
        context = ""
        if memory.task_results:
            context = (
                "Context from previous research steps:\n"
                + memory.get_context_summary()
            )

        # Set up messages for this task
        task_messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": TASK_EXECUTION_PROMPT.format(
                    query=task_query,
                    description=task_description,
                    context=context,
                ),
            },
        ]

        # ReAct loop
        for step in range(self.max_steps):
            step_start = time.time()

            response = self._client.chat.completions.create(
                model=self.model,
                messages=task_messages,
                tools=self._tools.to_openai_tools(),
                tool_choice="auto",
                temperature=0.1,
                max_tokens=settings.llm_max_tokens,
            )

            message = response.choices[0].message
            tokens_in = response.usage.prompt_tokens if response.usage else 0
            tokens_out = (
                response.usage.completion_tokens if response.usage else 0
            )
            step_end = time.time()

            # No tool calls — agent is done reasoning
            if not message.tool_calls:
                trace.add_step(
                    TraceStep(
                        step_id=f"reason_{step}",
                        step_type=StepType.LLM_CALL,
                        input_data={"task": task_query},
                        output_data={"response": message.content[:200]},
                        start_time=step_start,
                        end_time=step_end,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                    )
                )
                return message.content or "No answer generated."

            # Process tool calls
            task_messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                log.info(
                    "tool_call",
                    tool=tool_name,
                    args=str(tool_args)[:100],
                )

                # Execute the tool
                tool_result = self._tools.execute(tool_name, **tool_args)

                # Truncate very long results
                if len(tool_result) > 3000:
                    tool_result = tool_result[:3000] + "\n\n[Truncated]"

                # Add tool result to conversation
                task_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

                # Track source
                memory.add_source({
                    "tool": tool_name,
                    "query": tool_args.get("query", ""),
                    "result_preview": tool_result[:200],
                    "full_text": tool_result,
                })

                # Add to trace
                trace.add_step(
                    TraceStep(
                        step_id=f"tool_{tool_name}_{step}",
                        step_type=StepType.TOOL_USE,
                        input_data={
                            "tool": tool_name,
                            "args": tool_args,
                        },
                        output_data={
                            "result_length": len(tool_result),
                        },
                        start_time=step_start,
                        end_time=step_end,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                    )
                )

        # Max steps reached
        log.warning("max_steps_reached", task=task_query[:60])
        return "Max reasoning steps reached. Partial answer based on evidence gathered."

    def _synthesize(
        self,
        original_query: str,
        memory: AgentMemory,
        trace: Trace,
    ) -> str:
        """Synthesize results from all tasks into a final answer."""
        step_start = time.time()

        task_results_text = memory.get_context_summary()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SYNTHESIS_PROMPT.format(
                        original_query=original_query,
                        task_results=task_results_text,
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=settings.llm_max_tokens,
        )

        answer = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = (
            response.usage.completion_tokens if response.usage else 0
        )
        step_end = time.time()

        trace.add_step(
            TraceStep(
                step_id="synthesis",
                step_type=StepType.LLM_CALL,
                input_data={"original_query": original_query},
                output_data={"answer_length": len(answer or "")},
                start_time=step_start,
                end_time=step_end,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        )

        return answer or "Synthesis failed."
