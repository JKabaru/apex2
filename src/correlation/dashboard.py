from rich.console import Console
from rich.table import Table

console = Console()


def render_live_matrix(matrix_store) -> None:
    rows = matrix_store.get_latest_matrix()

    if not rows:
        console.print("[yellow]No correlation data yet — waiting for buffer to fill.[/]")
        return

    table = Table(title="LEW-CCF Correlation Matrix", border_style="dim")
    table.add_column("TF", style="bold cyan", no_wrap=True)
    table.add_column("Pair", style="bold")
    table.add_column("Coeff", justify="right")
    table.add_column("Lag", justify="right")
    table.add_column("Dir", justify="center")
    table.add_column("P-Value", justify="right")
    table.add_column("n_eff", justify="right")
    table.add_column("Status", justify="center")

    sorted_rows = sorted(rows, key=lambda r: (str(r.get("timeframe", "")), r["pair"]))

    for r in sorted_rows:
        sig = r["significant"]
        direction = r["direction"]
        style = (
            "bold green"
            if sig and direction > 0
            else "bold red" if sig and direction < 0
            else "dim white"
        )
        status = (
            "[green]SIG+[/]"
            if sig and direction > 0
            else "[red]SIG-[/]" if sig and direction < 0
            else "[dim]insig[/]"
        )

        table.add_row(
            r.get("timeframe", "1m"),
            r["pair"],
            f"{r['coefficient']:.4f}",
            str(r["dominant_lag"]),
            "+" if direction > 0 else "-",
            f"{r['p_value']:.4e}",
            f"{r['n_eff_joint']:.0f}",
            status,
            style=style,
        )

    console.print(table)
