"""Bulk ingest PDF files into the ATLAS vector store.

Usage:
    python scripts/ingest_papers.py --dir data/eval/
    python scripts/ingest_papers.py --file paper.pdf
"""

import argparse
import sys
from pathlib import Path
from time import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from atlas.retriever.ingest import DocumentIngestor

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into ATLAS")
    parser.add_argument("--dir", type=str, help="Directory containing PDF files")
    parser.add_argument("--file", type=str, help="Single PDF file to ingest")
    args = parser.parse_args()

    if not args.dir and not args.file:
        console.print("[red]Error: Provide --dir or --file[/red]")
        sys.exit(1)

    # Collect PDF files
    pdf_files: list[Path] = []
    if args.file:
        pdf_files.append(Path(args.file))
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.exists():
            console.print(f"[red]Directory not found: {dir_path}[/red]")
            sys.exit(1)
        pdf_files.extend(sorted(dir_path.glob("*.pdf")))

    if not pdf_files:
        console.print("[yellow]No PDF files found.[/yellow]")
        sys.exit(0)

    console.print(f"\n[bold]Found {len(pdf_files)} PDF file(s) to ingest[/bold]\n")

    # Ingest
    ingestor = DocumentIngestor()
    results = []

    for pdf in pdf_files:
        console.print(f"  Ingesting [cyan]{pdf.name}[/cyan]...", end=" ")
        start = time()
        try:
            result = ingestor.ingest(str(pdf))
            elapsed = time() - start
            results.append({
                "file": pdf.name,
                "doc_id": result["document_id"],
                "chunks": result["num_chunks"],
                "time": f"{elapsed:.1f}s",
                "status": "OK",
            })
            console.print(f"[green]{result['num_chunks']} chunks ({elapsed:.1f}s)[/green]")
        except Exception as e:
            elapsed = time() - start
            results.append({
                "file": pdf.name,
                "doc_id": "-",
                "chunks": 0,
                "time": f"{elapsed:.1f}s",
                "status": f"FAILED: {e}",
            })
            console.print(f"[red]FAILED: {e}[/red]")

    # Summary table
    console.print()
    table = Table(title="Ingestion Summary")
    table.add_column("File", style="cyan")
    table.add_column("Doc ID", style="dim")
    table.add_column("Chunks", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Status")

    for r in results:
        status_style = "green" if r["status"] == "OK" else "red"
        table.add_row(
            r["file"],
            r["doc_id"][:12] + "...",
            str(r["chunks"]),
            r["time"],
            f"[{status_style}]{r['status']}[/{status_style}]",
        )

    console.print(table)
    total_chunks = sum(r["chunks"] for r in results)
    console.print(
        f"\n[bold]Total: {len(results)} documents, {total_chunks} chunks, "
        f"{ingestor.dense_index.count} vectors in index[/bold]\n"
    )


if __name__ == "__main__":
    main()
