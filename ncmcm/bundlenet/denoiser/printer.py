from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
import numpy as np

# Console condivisa: stampe e barre di progresso non si sovrappongono
console = Console()

_STATUS_STYLES = {
    "success": ("✔", "green"),
    "warning": ("!", "yellow"),
    "error":   ("✘", "red"),
}


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[metrics]}"),
        console=console,
    )


def print_status(message: str, status: str = "success", details: dict | None = None) -> None:
    """
    Print a status message with an icon.

    Args:
        message: text to show.
        status: "success" (✔ green), "warning" (! yellow) or "error" (✘ red).
        details: optional {label: value}; if given, the message is shown
            in a panel with one row per item instead of a single line.
    """
    if status not in _STATUS_STYLES:
        raise ValueError(f"status must be one of {list(_STATUS_STYLES)}, got {status!r}")

    icon, color = _STATUS_STYLES[status]

    if not details:
        console.print(f"[bold {color}]{icon} {message}[/bold {color}]")
        return

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", justify="right")
    grid.add_column(style="bold")
    for label, value in details.items():
        grid.add_row(str(label), str(value))

    console.print()
    console.print(
        Panel(
            grid,
            title=f"[bold {color}]{icon} {message}[/bold {color}]",
            title_align="left",
            border_style=color,
            box=box.ROUNDED,
            padding=(1, 2),
            expand=False,
        )
    )
    console.print()


def print_success(message: str, details: dict | None = None) -> None:
    print_status(message, "success", details)


def print_warning(message: str, details: dict | None = None) -> None:
    print_status(message, "warning", details)


def print_error(message: str, details: dict | None = None) -> None:
    print_status(message, "error", details)

def make_table(
    title: str,
    columns: list,
    rows: list | None = None,
    *,
    title_style: str = "bold cyan",
    caption: str | None = None,
    box_style: box.Box = box.ROUNDED,
) -> Table:
    """
    Build a rich Table.

    Args:
        title: table title.
        columns: list of column specs. Each is either a string (the header)
            or a tuple (header, justify, style), e.g. ("Count", "right", "bold").
        rows: optional list of rows (iterables of values, converted to str).
            Rich markup is allowed, e.g. "[red]0[/red]".
        title_style: style of the title.
        caption: optional dim text under the table.
        box_style: rich box style.

    Returns:
        The Table, ready for ``console.print(table)``.
    """
    table = Table(
        title=title,
        title_style=title_style,
        box=box_style,
        header_style="bold white",
        caption=caption,
        caption_style="dim",
    )
    for col in columns:
        if isinstance(col, str):
            table.add_column(col)
        else:
            header, justify, *rest = col
            table.add_column(header, justify=justify, style=rest[0] if rest else None)

    for row in rows or []:
        table.add_row(*(str(v) for v in row))

    return table


def print_split_summary(B, train_indices, test_indices, title: str = "Splits per behavioral class") -> None:
    """Print train/test sample counts (and percentages) per behavioral class."""
    train_labels = np.asarray(B)[train_indices]
    test_labels = np.asarray(B)[test_indices]

    classes = np.union1d(train_labels, test_labels)
    n_train, n_test = len(train_labels), len(test_labels)

    def _count(labels, c):
        return int(np.sum(labels == c))

    def _cell(n, total):
        if n == 0:
            return "[red]0[/red]", "[red]0.0%[/red]"
        return str(n), f"{100 * n / total:.1f}%"

    rows = []
    for c in classes:
        tr, tr_pct = _cell(_count(train_labels, c), n_train)
        te, te_pct = _cell(_count(test_labels, c), n_test)
        rows.append([f"[bold]{c}[/bold]", tr, tr_pct, te, te_pct])

    table = make_table(
        title,
        columns=[
            "Class",
            ("Train", "right"),
            ("Train %", "right", "dim"),
            ("Test", "right"),
            ("Test %", "right", "dim"),
        ],
        rows=rows,
        caption="Red 0 = class absent from that split",
    )
    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{n_train}[/bold]", "", f"[bold]{n_test}[/bold]", "")

    console.print()
    console.print(table)
    console.print()

def print_hidden_layers(den_hidden_layers, title: str = "Hidden layer size for the denoiser") -> None:
    """Print the hidden layer sizes as a table, plus a one-line architecture summary."""
    sizes = [int(s) for s in den_hidden_layers]

    rows = [[f"[dim]{i}[/dim]", f"[bold]{s}[/bold]"] for i, s in enumerate(sizes, start=1)]

    table = make_table(
        title,
        columns=[("Layer", "right"), ("Units", "right")],
        rows=rows,
    )

    console.print()
    console.print(table)
    console.print(
        "[dim]Architecture:[/dim] "
        + " [cyan]→[/cyan] ".join(f"[bold]{s}[/bold]" for s in sizes)
        + f"   [dim]({len(sizes)} hidden layers, {sum(sizes)} units total)[/dim]"
    )
    console.print()