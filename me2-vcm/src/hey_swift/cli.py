"""Command-line interface for Hey Swift.

Entry point: `hey-swift` (see [project.scripts] in pyproject.toml) or
`python -m hey_swift.cli`.

Commands:
    train <wakeword|vcm> [--tag TAG] [--config PATH]
    export onnx --tag TAG
    run                 # live: wake -> command -> action
    eval <vcm> --tag TAG
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="hey-swift",
    help="Hey Swift wake-word + Voice Command Model pipeline.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def train(
    target: str = typer.Argument(..., help="'wakeword' or 'vcm'."),
    tag: str = typer.Option("a0_scratch", "--tag", "-t", help="Experiment tag."),
    config: str = typer.Option("configs/vcm.yaml", "--config", "-c", help="Config YAML."),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed."),
) -> None:
    """Train a model. Writes immutable artifacts under experiments/<target>/<tag>/."""
    if target == "vcm":
        from hey_swift.vcm.train import run as run_train
    elif target == "wakeword":
        from hey_swift.wakeword.train import run as run_train
    else:
        console.print(f"[red]Unknown target:[/red] {target}")
        raise typer.Exit(code=2)
    run_train(tag=tag, config=config, seed=seed)


@app.command()
def export(
    fmt: str = typer.Argument("onnx", help="Export format ('onnx')."),
    tag: str = typer.Option("a0_scratch", "--tag", "-t"),
) -> None:
    """Export a trained model to ONNX for CPU deployment."""
    from hey_swift.vcm.inference import export_onnx
    export_onnx(tag=tag)


@app.command()
def run() -> None:
    """Run the live end-to-end pipeline: wake -> command -> action."""
    from hey_swift.app import main
    main()


@app.command()
def eval_(
    target: str = typer.Argument("vcm", help="'vcm' (or 'wakeword')."),
    tag: str = typer.Option("a0_scratch", "--tag", "-t"),
) -> None:
    """Evaluate a trained model on the frozen test / holdout splits."""
    from hey_swift.vcm.evaluate import run as run_eval
    run_eval(tag=tag)


if __name__ == "__main__":
    app()
