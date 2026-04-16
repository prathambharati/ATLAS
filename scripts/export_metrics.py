"""Export evaluation metrics as a markdown summary.

Reads results.json from the evaluation run and generates
a clean markdown report suitable for the README or docs.

Usage:
    python scripts/export_metrics.py
    python scripts/export_metrics.py --input data/eval/results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()


def export_markdown(results_path: str, output_path: str) -> None:
    """Generate markdown metrics summary from results JSON."""
    with open(results_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    successful = [r for r in results if r["status"] == "success"]

    if not successful:
        console.print("[red]No successful results to export.[/red]")
        return

    # Calculate metrics
    total = len(results)
    success_count = len(successful)
    avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
    avg_latency = sum(r["latency_s"] for r in successful) / len(successful)
    total_claims = sum(r["total_claims"] for r in successful)
    supported = sum(r["supported_claims"] for r in successful)
    groundedness = supported / total_claims if total_claims > 0 else 0

    # Per-complexity breakdown
    simple = [r for r in successful if r["complexity"] == "simple"]
    complex_q = [r for r in successful if r["complexity"] == "complex"]

    simple_conf = (
        sum(r["confidence"] for r in simple) / len(simple) if simple else 0
    )
    complex_conf = (
        sum(r["confidence"] for r in complex_q) / len(complex_q)
        if complex_q
        else 0
    )

    # Build markdown
    md = []
    md.append("## Evaluation Results\n")
    md.append(
        f"Evaluated **{total}** research questions "
        f"({success_count} successful).\n"
    )

    md.append("### Overall Metrics\n")
    md.append("| Metric | Value | Target |")
    md.append("|--------|-------|--------|")
    md.append(f"| Groundedness | **{groundedness:.1%}** | ≥ 85% |")
    md.append(f"| Avg Confidence | **{avg_confidence:.1%}** | ≥ 70% |")
    md.append(f"| Total Claims Verified | **{total_claims}** | — |")
    md.append(f"| Claims Supported | **{supported}** | — |")
    md.append(f"| Avg Latency | **{avg_latency:.1f}s** | < 60s |")
    md.append(f"| Success Rate | **{success_count}/{total}** | 100% |")

    md.append("\n### By Complexity\n")
    md.append("| Type | Count | Avg Confidence |")
    md.append("|------|-------|----------------|")
    md.append(f"| Simple | {len(simple)} | {simple_conf:.1%} |")
    md.append(f"| Complex | {len(complex_q)} | {complex_conf:.1%} |")

    md.append("\n### Per-Question Results\n")
    md.append("| ID | Query | Confidence | Claims | Latency |")
    md.append("|-----|-------|------------|--------|---------|")

    for r in successful:
        conf = r["confidence"]
        icon = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.5 else "🔴"
        md.append(
            f"| {r['id']} | {r['query'][:35]}... "
            f"| {icon} {conf:.0%} "
            f"| {r['total_claims']} "
            f"| {r['latency_s']}s |"
        )

    output = "\n".join(md)

    with open(output_path, "w") as f:
        f.write(output)

    console.print(f"[green]Metrics exported to {output_path}[/green]")
    console.print()
    console.print(output)


def main():
    parser = argparse.ArgumentParser(
        description="Export evaluation metrics as markdown"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/eval/results.json",
        help="Path to results JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/EVALUATION.md",
        help="Output markdown path",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        console.print(
            f"[red]Results file not found: {args.input}[/red]\n"
            f"Run 'python scripts/run_eval.py' first."
        )
        sys.exit(1)

    export_markdown(args.input, args.output)


if __name__ == "__main__":
    main()
