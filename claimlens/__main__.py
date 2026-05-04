from __future__ import annotations

import argparse
import sys
import textwrap

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from claimlens import pipeline
from claimlens.models import ClaimVerdict

_VERDICT_COLOUR = {
    ClaimVerdict.SUPPORTED: "green",
    ClaimVerdict.REFUTED: "red",
    ClaimVerdict.INSUFFICIENT_EVIDENCE: "yellow",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="claimlens",
        description="Verify a claim against live web evidence.",
    )
    parser.add_argument("claim", nargs="+", help="Claim to verify (no quotes needed)")
    args = parser.parse_args()

    claim = " ".join(args.claim)

    console = Console()

    with console.status("Verifying…"):
        result = pipeline.run(claim)

    colour = _VERDICT_COLOUR[result.verdict]
    verdict_str = f"[bold {colour}]{result.verdict.value}[/bold {colour}]"
    confidence_str = f"Confidence: [bold]{result.confidence * 100:.1f}%[/bold]"

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=3)
    table.add_column("Label", width=10)
    table.add_column("Source", overflow="fold", max_width=40)
    table.add_column("Quote", overflow="fold", max_width=60)

    label_colours = {"SUPPORTS": "green", "REFUTES": "red", "NEUTRAL": "yellow"}
    for i, ev in enumerate(result.evidence[:3], start=1):
        label_val = ev.label.value
        label_col = label_colours.get(label_val, "white")
        quote = textwrap.shorten(ev.chunk_text, width=120, placeholder="…")
        table.add_row(
            str(i),
            f"[{label_col}]{label_val}[/{label_col}]",
            ev.source_url,
            quote,
        )

    panel_content = f"{verdict_str}\n{confidence_str}\n\n"
    console.print(
        Panel(
            panel_content,
            title="[bold]ClaimLens Verdict[/bold]",
            border_style=colour,
        )
    )
    console.print(table)


if __name__ == "__main__":
    main()
