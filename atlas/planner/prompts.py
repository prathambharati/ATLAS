"""Prompt templates for the query decomposition planner.

These prompts instruct the LLM to break a complex research question
into a DAG of sub-tasks with explicit dependencies.
"""

DECOMPOSER_SYSTEM_PROMPT = """Your job is to decompose complex research questions into smaller,
focused sub-questions that can be answered independently or in sequence.

Rules:
1. Break the query into 2-3 sub-tasks (no more than 3).
2. Each sub-task should be a focused, answerable question.
3. Identify dependencies: if answering task B requires the result of task A, mark it.
4. Tasks with no dependencies can be executed in parallel.
5. The final task should synthesize the results of previous tasks.
6. Keep sub-tasks specific and actionable.

Respond ONLY with valid JSON in this exact format, no extra text:

{
  "tasks": [
    {
      "id": "t1",
      "query": "What is X?",
      "description": "Brief description of what this task does",
      "depends_on": []
    },
    {
      "id": "t2",
      "query": "What is Y?",
      "description": "Brief description",
      "depends_on": []
    },
    {
      "id": "t3",
      "query": "How do X and Y compare?",
      "description": "Synthesize findings from t1 and t2",
      "depends_on": ["t1", "t2"]
    }
  ]
}"""

DECOMPOSER_USER_PROMPT = """Decompose this research question into sub-tasks:

Research Question: {query}

Remember:
- 2-6 sub-tasks maximum
- Include dependencies between tasks
- Final task should synthesize previous results
- Respond with JSON only"""


SIMPLE_QUERY_SYSTEM_PROMPT = """Determine if a research question is simple
(can be answered directly with a single search) or complex
(needs to be broken into sub-questions).

A query is SIMPLE if:
- It asks about a single concept or fact
- It can be answered with one search
- Examples: "What is BERT?", "When was GPT-3 released?"

A query is COMPLEX if:
- It asks for comparison, analysis, or multi-step reasoning
- It references multiple concepts that need separate investigation
- It asks "how" or "why" with multiple factors
- Examples: "- Examples: "Compare BERT vs GPT-2 for text classification",
  "What are the tradeoffs between speculative decoding and continuous batching?"

Respond with ONLY "simple" or "complex". Nothing else."""
