"""Gera o gráfico de comparação entre gradient descent e conjugate gradient."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "results" / "tables" / "optimizer_comparison.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "figures" / "optimizer_comparison.png"
REQUIRED_COLUMNS = {
    "iterations",
    "lambda",
    "gradient_descent_validation_accuracy",
    "conjugate_gradient_validation_accuracy",
}


def plot_optimizer_comparison(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    show: bool = False,
) -> Path:
    """Lê a tabela de experimentos e salva a comparação dos otimizadores."""
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Tabela não encontrada: {csv_path}. "
            "Execute `uv run python experiments/generate_report.py` primeiro."
        )

    data = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Tabela sem as colunas esperadas: {sorted(missing_columns)}")

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    configurations = (
        ("gradient_descent_validation_accuracy", "Gradiente descendente", "o"),
        ("conjugate_gradient_validation_accuracy", "Gradiente conjugado", "s"),
    )
    for axis, (column, title, marker) in zip(axes, configurations):
        for lambda_value, subset in data.groupby("lambda", sort=True):
            subset = subset.sort_values("iterations")
            axis.plot(
                subset["iterations"],
                subset[column],
                marker=marker,
                label=f"λ = {lambda_value:g}",
            )
        axis.set_title(title)
        axis.set_xlabel("Número de iterações")
        axis.grid(linestyle="--", alpha=0.6)
        axis.legend(loc="lower right", fontsize=9, framealpha=0.8)
    axes[0].set_ylabel("Acurácia de validação (%)")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera o gráfico a partir de optimizer_comparison.csv."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_optimizer_comparison(args.input, args.output, args.show)
    print(f"Gráfico salvo em: {output_path}")


if __name__ == "__main__":
    main()
