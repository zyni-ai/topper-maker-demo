"""CLI entry point — `uv run topper-maker` or `python run_demo.py`.

Looks for Physics_Model_Papers_Answer_Key.pdf and ground_truth/ relative to the
current working directory (i.e. the project root). Override with --pdf / --gt.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

_DEFAULT_PDF = "Physics_Model_Papers_Answer_Key.pdf"
_DEFAULT_GT = "ground_truth/pages_01_03.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topper Maker HTR demo")
    parser.add_argument("--pages", type=int, default=3, help="Number of pages (0 = all)")
    parser.add_argument("--engine", choices=["openrouter", "azure"], default="openrouter")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pdf", default=None, help="Path to answer-sheet PDF")
    parser.add_argument("--gt", default=None, help="Path to ground-truth JSON")
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    from topper_maker.pipeline import ExtractionPipeline, PipelineConfig

    console = Console(highlight=False, emoji=False, markup=True)
    pdf_path = Path(args.pdf) if args.pdf else Path.cwd() / _DEFAULT_PDF
    gt_path = Path(args.gt) if args.gt else Path.cwd() / _DEFAULT_GT

    if not pdf_path.exists():
        console.print(f"[red]PDF not found: {pdf_path}[/red]")
        sys.exit(1)

    config = PipelineConfig.from_env()
    config.htr_engine = args.engine
    config.max_pages = args.pages
    config.target_dpi = args.dpi
    config.ground_truth_path = str(gt_path) if gt_path.exists() else None

    console.rule("[bold cyan]Topper Maker — HTR Extraction Pipeline[/bold cyan]")
    console.print(f"  PDF     : {pdf_path.name}")
    console.print(f"  Engine  : [bold]{config.htr_engine}[/bold]")
    console.print(f"  Pages   : {config.max_pages}")
    console.print(f"  DPI     : {config.target_dpi}")
    console.print()

    pipeline = ExtractionPipeline(config)
    result = pipeline.run(pdf_path)

    # Quality reports
    console.rule("[yellow]Stage 2 — Image Quality[/yellow]")
    q_table = Table(box=box.SIMPLE)
    q_table.add_column("Page", style="cyan")
    q_table.add_column("DPI")
    q_table.add_column("Blur Score")
    q_table.add_column("Brightness")
    q_table.add_column("Contrast")
    q_table.add_column("Status")
    for r in result.quality_reports:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        q_table.add_row(
            str(r.page_number), str(r.dpi),
            f"{r.blur_score:.1f}", f"{r.brightness:.1f}", f"{r.contrast:.1f}", status,
        )
    console.print(q_table)

    # HTR output
    console.rule("[yellow]Stage 4 — HTR Extracted Text[/yellow]")
    for htr in result.htr_results:
        console.print(
            f"\n[bold]Page {htr.page_number}[/bold] "
            f"({len(htr.blocks)} blocks, mean_conf={htr.mean_confidence:.3f})"
        )
        console.print("-" * 60)
        for block in htr.blocks[:10]:
            conf_color = "green" if block.confidence >= 0.85 else "yellow" if block.confidence >= 0.70 else "red"
            console.print(
                f"  [{conf_color}]{block.confidence:.2f}[/{conf_color}]  "
                f"[{block.region_type.value}]  {block.text[:120]}"
            )
        if len(htr.blocks) > 10:
            console.print(f"  ... ({len(htr.blocks) - 10} more blocks)")

    # Document structure
    console.rule("[yellow]Stage 5 — Detected Structure[/yellow]")
    if result.structure:
        s = result.structure
        console.print(f"  Parts detected      : {s.parts}")
        console.print(f"  Sections detected   : {s.sections}")
        console.print(f"  Questions detected  : {s.question_numbers}")
        console.print(f"  Total unique Q's    : {s.total_questions_detected}")

    # Metrics
    console.rule("[yellow]Stage 6 — Extraction Metrics[/yellow]")
    if result.report:
        report = result.report
        m_table = Table(box=box.SIMPLE)
        m_table.add_column("Page", style="cyan")
        m_table.add_column("CER (low=good)", justify="right")
        m_table.add_column("WER (low=good)", justify="right")
        m_table.add_column("Mean Conf", justify="right")
        m_table.add_column("Min Conf", justify="right")
        m_table.add_column("Blocks")
        m_table.add_column("Review?")
        for pm in report.page_metrics:
            gt_note = "" if config.ground_truth_path else " (est.)"
            review = "[red]YES[/red]" if pm.needs_human_review else "[green]no[/green]"
            m_table.add_row(
                str(pm.page_number),
                f"{pm.cer:.4f}{gt_note}", f"{pm.wer:.4f}{gt_note}",
                f"{pm.mean_confidence:.4f}", f"{pm.min_confidence:.4f}",
                str(pm.num_blocks), review,
            )
        console.print(m_table)

        summary = report.summary()
        console.print("\n[bold]Aggregate:[/bold]")
        console.print(f"  Mean CER           : {summary['mean_cer']:.4f}")
        console.print(f"  Mean WER           : {summary['mean_wer']:.4f}")
        console.print(f"  Mean Confidence    : {summary['mean_confidence']:.4f}")
        console.print(f"  Pages flagged      : {summary['pages_flagged_for_review']}")

        if summary.get("structure"):
            st = summary["structure"]
            console.print("\n[bold]Structure Detection:[/bold]")
            console.print(f"  Question recall    : {st['question_recall']:.4f}")
            console.print(f"  Question precision : {st['question_precision']:.4f}")
            console.print(f"  Part recall        : {st['part_recall']:.4f}")
            console.print(f"  Section recall     : {st['section_recall']:.4f}")

    if result.errors:
        console.rule("[red]Errors[/red]")
        for e in result.errors:
            console.print(f"  [red]x[/red] {e}")

    console.rule("[bold green]Done[/bold green]")


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    # Force UTF-8 on Windows before any output (avoids cp1252 crash on math chars)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _run(_parse_args())
