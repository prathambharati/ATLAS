"""Run evaluation suite against the ATLAS agent.

Loads questions from data/eval/questions.json, runs each through the agent,
evaluates with the hallucination detector, and produces a results summary.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --max-questions 5
    python scripts/run_eval.py --simple-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from atlas.agent.orchestrator import AgentOrchestrator
from atlas.config import settings
from atlas.evaluator.evaluator import HallucinationEvaluator

console = Console()


def load_questions(
    path: str,
    max_questions: int | None = None,
    complexity: str | None = None,
) -> list[dict]:
    """Load evaluation questions from JSON file."""
    with open(path) as f:
        data = json.load(f)

    questions = data.get("eval_set", [])

    if complexity:
        questions = [q for q in questions if q.get("complexity") == complexity]

    if max_questions:
        questions = questions[:max_questions]

    return questions


def evaluate_answer(
    evaluator: HallucinationEvaluator,
    answer: str,
    sources: list[dict],
) -> dict:
    """Run hallucination evaluation on an answer."""
    evidence_chunks = []
    for source in sources:
        text = source.get("full_text", source.get("result_preview", ""))
        if text and "Error" not in text:
            evidence_chunks.append({
                "text": text[:1000],
                "source": source.get("tool", "unknown"),
            })

    if not evidence_chunks:
        return {
            "total_claims": 0,
            "supported": 0,
            "unsupported": 0,
            "contradicted": 0,
            "confidence": 0.0,
        }

    report = evaluator.evaluate(answer, evidence_chunks)
    return {
        "total_claims": report.total_claims,
        "supported": report.supported_claims,
        "unsupported": report.unsupported_claims,
        "contradicted": report.contradicted_claims,
        "confidence": report.overall_confidence,
    }


def run_evaluation(
    questions: list[dict],
    agent: AgentOrchestrator,
    evaluator: HallucinationEvaluator,
) -> list[dict]:
    """Run all evaluation questions through the agent."""
    results = []

    for i, q in enumerate(questions, 1):
        console.print(
            f"\n[bold cyan]Question {i}/{len(questions)}:[/bold cyan] "
            f"{q['query'][:80]}"
        )
        console.print(f"  Complexity: {q['complexity']} | Domain: {q['domain']}")

        start = time.time()
        try:
            agent_result = agent.run(q["query"])
            elapsed = time.time() - start

            answer = agent_result.get("answer", "")
            sources = agent_result.get("sources", [])
            dag = agent_result.get("dag_summary", {})

            # Evaluate for hallucinations
            eval_result = evaluate_answer(evaluator, answer, sources)

            result = {
                "id": q["id"],
                "query": q["query"],
                "complexity": q["complexity"],
                "domain": q["domain"],
                "answer_length": len(answer),
                "num_sources": len(sources),
                "num_tasks": dag.get("num_tasks", 1),
                "latency_s": round(elapsed, 1),
                "confidence": eval_result["confidence"],
                "total_claims": eval_result["total_claims"],
                "supported_claims": eval_result["supported"],
                "unsupported_claims": eval_result["unsupported"],
                "status": "success",
                "error": None,
            }

            conf = eval_result["confidence"]
            icon = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.5 else "🔴"
            console.print(
                f"  {icon} Confidence: {conf:.0%} | "
                f"Claims: {eval_result['total_claims']} | "
                f"Supported: {eval_result['supported']} | "
                f"Time: {elapsed:.1f}s"
            )

        except Exception as e:
            elapsed = time.time() - start
            result = {
                "id": q["id"],
                "query": q["query"],
                "complexity": q["complexity"],
                "domain": q["domain"],
                "answer_length": 0,
                "num_sources": 0,
                "num_tasks": 0,
                "latency_s": round(elapsed, 1),
                "confidence": 0.0,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "status": "failed",
                "error": str(e),
            }
            console.print(f"  [red]FAILED: {e}[/red]")

        results.append(result)

    return results


def print_summary(results: list[dict]) -> None:
    """Print a summary table of evaluation results."""
    console.print("\n")

    # Results table
    table = Table(title="ATLAS Evaluation Results")
    table.add_column("ID", style="dim")
    table.add_column("Query", max_width=40)
    table.add_column("Type", style="cyan")
    table.add_column("Confidence", justify="right")
    table.add_column("Claims", justify="right")
    table.add_column("Supported", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Status")

    for r in results:
        conf = r["confidence"]
        if r["status"] == "failed":
            conf_str = "[red]FAIL[/red]"
            status = "[red]❌[/red]"
        elif conf >= 0.8:
            conf_str = f"[green]{conf:.0%}[/green]"
            status = "[green]✅[/green]"
        elif conf >= 0.5:
            conf_str = f"[yellow]{conf:.0%}[/yellow]"
            status = "[yellow]⚠️[/yellow]"
        else:
            conf_str = f"[red]{conf:.0%}[/red]"
            status = "[red]🔴[/red]"

        table.add_row(
            r["id"],
            r["query"][:40],
            r["complexity"],
            conf_str,
            str(r["total_claims"]),
            str(r["supported_claims"]),
            f"{r['latency_s']}s",
            status,
        )

    console.print(table)

    # Aggregate metrics
    successful = [r for r in results if r["status"] == "success"]
    if not successful:
        console.print("[red]No successful evaluations.[/red]")
        return

    avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
    avg_latency = sum(r["latency_s"] for r in successful) / len(successful)
    total_claims = sum(r["total_claims"] for r in successful)
    total_supported = sum(r["supported_claims"] for r in successful)
    groundedness = total_supported / total_claims if total_claims > 0 else 0

    metrics_table = Table(title="Aggregate Metrics")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Value", justify="right")

    metrics_table.add_row("Questions Evaluated", str(len(results)))
    metrics_table.add_row("Successful", str(len(successful)))
    metrics_table.add_row("Failed", str(len(results) - len(successful)))
    metrics_table.add_row("Avg Confidence", f"{avg_confidence:.1%}")
    metrics_table.add_row("Groundedness", f"{groundedness:.1%}")
    metrics_table.add_row("Total Claims", str(total_claims))
    metrics_table.add_row("Claims Supported", str(total_supported))
    metrics_table.add_row("Avg Latency", f"{avg_latency:.1f}s")

    console.print(metrics_table)


def save_results(results: list[dict], output_path: str) -> None:
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    console.print(f"\n[green]Results saved to {output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Run ATLAS evaluation suite")
    parser.add_argument(
        "--questions",
        type=str,
        default="data/eval/questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Max number of questions to evaluate",
    )
    parser.add_argument(
        "--simple-only",
        action="store_true",
        help="Only run simple (non-decomposed) questions",
    )
    parser.add_argument(
        "--complex-only",
        action="store_true",
        help="Only run complex (decomposed) questions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval/results.json",
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    # Validate
    if not Path(args.questions).exists():
        console.print(f"[red]Questions file not found: {args.questions}[/red]")
        sys.exit(1)

    if not settings.openai_api_key:
        console.print("[red]OPENAI_API_KEY not set in .env[/red]")
        sys.exit(1)

    # Load questions
    complexity = None
    if args.simple_only:
        complexity = "simple"
    elif args.complex_only:
        complexity = "complex"

    questions = load_questions(
        args.questions,
        max_questions=args.max_questions,
        complexity=complexity,
    )

    console.print(
        f"\n[bold]ATLAS Evaluation Suite[/bold]\n"
        f"Questions: {len(questions)} | "
        f"Model: {settings.llm_model}\n"
    )

    # Initialize components
    console.print("Loading agent...", end=" ")
    agent = AgentOrchestrator(max_steps=8)
    console.print("[green]ready[/green]")

    console.print("Loading evaluator...", end=" ")
    evaluator = HallucinationEvaluator()
    console.print("[green]ready[/green]")

    # Run evaluation
    results = run_evaluation(questions, agent, evaluator)

    # Print summary
    print_summary(results)

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
