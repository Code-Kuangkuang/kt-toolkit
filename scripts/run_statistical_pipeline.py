#!/usr/bin/env python
"""Run the complete statistical testing pipeline for a removed model.

This script automates:
1. Export predictions for all datasets and models
2. Run statistical tests
3. Apply Holm-Bonferroni correction
4. Generate summary report

Usage:
    python scripts/run_statistical_pipeline.py --step export
    python scripts/run_statistical_pipeline.py --step test
    python scripts/run_statistical_pipeline.py --step all
"""

import os
import subprocess
import sys
from pathlib import Path

import typer
from rich import print

# Get project root directory
ROOT = Path(__file__).resolve().parents[1]

app = typer.Typer(add_completion=False)

# Dataset-baseline mapping as specified by user
DATASET_BASELINE_MAP = {
    "algebra2005": "ukt",
    "assist2009": "ukt",
    "assist2017": "ukt",
    "bridge2algebra2006": "akt",
    "nips_task34": "qikt",
}

DATASETS = list(DATASET_BASELINE_MAP.keys())
MODEL_A = "removed_model"


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n[blue]Running: {description}[/blue]")
    print(f"[dim]Command: {' '.join(cmd)}[/dim]\n")

    result = subprocess.run(cmd, capture_output=False, cwd=str(ROOT))

    if result.returncode != 0:
        print(f"[red]Error running: {description}[/red]")
        return False

    print(f"[green]Completed: {description}[/green]")
    return True


@app.command()
def main(
    step: str = typer.Option("all", "--step", help="Which step to run: export, test, or all"),
    datasets: str = typer.Option(
        "algebra2005,assist2009,assist2017,bridge2algebra2006,nips_task34",
        "--datasets",
        help="Comma-separated dataset names"
    ),
    n_permutations: int = typer.Option(10000, "--n-perm", help="Number of permutations"),
    n_bootstrap: int = typer.Option(10000, "--n-boot", help="Number of bootstrap samples"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    device: str = typer.Option("cuda:0", "--device", help="Device for inference"),
):
    """Run the complete statistical testing pipeline."""

    dataset_list = [d.strip() for d in datasets.split(",")]

    if step in ["export", "all"]:
        print("\n" + "=" * 70)
        print("  STEP 1: Export predictions for all models")
        print("=" * 70)

        for dataset in dataset_list:
            baseline = DATASET_BASELINE_MAP.get(dataset, "ukt")

            # Export a removed model predictions
            success = run_command(
                ["conda", "run", "-n", "pykt312", "python", "scripts/export_predictions.py",
                 "-d", dataset, "-m", MODEL_A, "--device", device],
                f"Export {MODEL_A} predictions for {dataset}"
            )

            # Export baseline predictions
            success = run_command(
                ["conda", "run", "-n", "pykt312", "python", "scripts/export_predictions.py",
                 "-d", dataset, "-m", baseline, "--device", device],
                f"Export {baseline} predictions for {dataset}"
            )

    if step in ["test", "all"]:
        print("\n" + "=" * 70)
        print("  STEP 2: Run statistical tests with Holm-Bonferroni correction")
        print("=" * 70)

        success = run_command(
            ["conda", "run", "-n", "pykt312", "python", "scripts/statistical_test.py", "batch",
             "--datasets", datasets,
             "--model-a", MODEL_A,
             "--n-perm", str(n_permutations),
             "--n-boot", str(n_bootstrap),
             "--seed", str(seed),
             "--holm"],
            "Run batch statistical tests"
        )

    if step == "all":
        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETE")
        print("=" * 70)
        print("\n[green]Results saved to:[/green]")
        print("  - output/statistical_tests/statistical_test_summary.json")
        print("  - output/statistical_tests/statistical_test_summary.md")


@app.command("quick")
def quick_test(
    datasets: str = typer.Option("algebra2005", "--datasets", help="Datasets for quick test"),
    n_permutations: int = typer.Option(1000, "--n-perm", help="Number of permutations (reduced for quick test)"),
    n_bootstrap: int = typer.Option(1000, "--n-boot", help="Number of bootstrap samples (reduced for quick test)"),
):
    """Quick test with reduced iterations for verification."""

    print("[yellow]Running quick test with reduced iterations (1000 each)[/yellow]")

    main(
        step="all",
        datasets=datasets,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
    )


if __name__ == "__main__":
    app()