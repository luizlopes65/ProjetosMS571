"""Executa os experimentos e gera um relatório final reproduzível.

Exemplo rápido:
    python experiments/generate_report.py --iterations 2 --optimizer-iterations 2

Execução completa:
    python experiments/generate_report.py
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.conjugate_gradients import compare_optimizers, export_comparison_table
from experiments.error_cases_viz import run_error_cases_visualization
from experiments.gradient_check import gradient_check
from experiments.lambda_search import LAMBDA_LIST, plot_validation_error, search_lambda
from experiments.learning_curve import run_learning_curve
from experiments.visualize_neurons import create_activation_gif
from train import get_data_partitioned, predict_model, train_model

DEFAULT_DATA_PATH = PROJECT_ROOT / "processed_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"
DEFAULT_HIDDEN_LAYER_SIZES = [25]
DEFAULT_ITERATIONS = 800
DEFAULT_OPTIMIZER_ITERATIONS = (50, 100, 200, 400, 800)
DEFAULT_RANDOM_STATE = 31


@contextmanager
def preserve_random_state():
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa os experimentos e gera figuras, tabelas e relatório."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--optimizer-iterations",
        type=int,
        nargs="+",
        default=list(DEFAULT_OPTIMIZER_ITERATIONS),
        metavar="N",
    )
    parser.add_argument(
        "--hidden-layer-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_LAYER_SIZES,
        metavar="N",
    )
    return parser.parse_args()


def _validate_args(iterations, optimizer_iterations, hidden_layer_sizes):
    if iterations <= 0:
        raise ValueError("iterations deve ser positivo.")
    if not optimizer_iterations or any(value <= 0 for value in optimizer_iterations):
        raise ValueError("optimizer_iterations deve conter valores positivos.")
    if not hidden_layer_sizes or any(value <= 0 for value in hidden_layer_sizes):
        raise ValueError("hidden_layer_sizes deve conter valores positivos.")


def _run_gradient_checks(seed):
    checks = []
    for architecture in ([3, 3], [3, 5, 3], [3, 5, 4, 3]):
        with redirect_stdout(StringIO()):
            passed = gradient_check(architecture, seed=seed)
        checks.append({"architecture": architecture, "passed": bool(passed)})
    return checks


def _write_gradient_checks(checks, output_path):
    output_path.write_text(
        "| Arquitetura | Resultado |\n| --- | --- |\n"
        + "\n".join(
            f"| `{'-'.join(map(str, item['architecture']))}` | "
            f"{'OK' if item['passed'] else 'FALHOU'} |"
            for item in checks
        )
        + "\n",
        encoding="utf-8",
    )


def _write_report(report_path, config, checks, lambda_result, learning_result, optimizer_results, test_accuracy):
    best_gd = max(optimizer_results, key=lambda row: row["gradient_descent_accuracy"])
    best_cg = max(optimizer_results, key=lambda row: row["conjugate_gradient_accuracy"])
    lines = [
        "# Relatório de experimentos",
        "",
        "## Configuração",
        "",
        f"- Dados: `{config['data_path']}`",
        f"- Seed: `{config['seed']}`",
        f"- Arquitetura: `[{config['input_size']}, {', '.join(map(str, config['hidden_layer_sizes']))}, {config['num_labels']}]`",
        f"- Iterações principais: `{config['iterations']}`",
        "",
        "## Gradient check",
        "",
        "| Arquitetura | Resultado |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{'-'.join(map(str, item['architecture']))}` | "
        f"{'OK' if item['passed'] else 'FALHOU'} |"
        for item in checks
    )
    lines.extend(
        [
            "",
            "## Busca de lambda",
            "",
            f"- Melhor lambda: `{lambda_result['best_lambda']}`",
            f"- Acurácia de validação: `{lambda_result['validation_accuracy']:.2f}%`",
            f"- Acurácia de teste: `{test_accuracy:.2f}%`",
            "",
            "## Curva de aprendizado",
            "",
            f"- Menor erro de treino: `{learning_result['train_errors'].min():.2f}%`",
            f"- Menor erro de validação: `{learning_result['validation_errors'].min():.2f}%`",
            "",
            "## Comparação entre otimizadores",
            "",
            f"- Melhor gradient descent: `{best_gd['gradient_descent_accuracy']:.2f}%` "
            f"(iterações={best_gd['iterations']}, lambda={best_gd['lambda']})",
            f"- Melhor conjugate gradient: `{best_cg['conjugate_gradient_accuracy']:.2f}%` "
            f"(iterações={best_cg['iterations']}, lambda={best_cg['lambda']})",
            "",
            "## Arquivos gerados",
            "",
            "### Figuras",
            "",
            "- [Busca de lambda](figures/lambda_search_costs.png)",
            "- [Erro de validação por lambda](figures/validation_error_vs_lambda.png)",
            "- [Curva de aprendizado](figures/learning_curve.png)",
            "- [Custo por tamanho de treino](figures/learning_curve_by_train_size.png)",
            "- [Evolução dos pesos](figures/activation_evolution.gif)",
            "- [Casos classificados incorretamente](figures/error_cases.png)",
            "",
            "### Tabelas",
            "",
            "- [Comparação dos otimizadores em CSV](tables/optimizer_comparison.csv)",
            "- [Comparação dos otimizadores em Markdown](tables/optimizer_comparison.md)",
            "- [Resultados do gradient check](tables/gradient_check.md)",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(
    data_path=DEFAULT_DATA_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    seed=DEFAULT_RANDOM_STATE,
    iterations=DEFAULT_ITERATIONS,
    optimizer_iterations=DEFAULT_OPTIMIZER_ITERATIONS,
    hidden_layer_sizes=None,
):
    if hidden_layer_sizes is None:
        hidden_layer_sizes = list(DEFAULT_HIDDEN_LAYER_SIZES)
    optimizer_iterations = tuple(optimizer_iterations)
    _validate_args(iterations, optimizer_iterations, hidden_layer_sizes)

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data_partitioned(
        file_path=data_path,
        random_state=seed,
    )
    input_size = X_train.shape[1]
    num_labels = int(
        max(
            np.asarray(y_train).max(),
            np.asarray(y_valid).max(),
            np.asarray(y_test).max(),
        )
    )

    checks = _run_gradient_checks(seed)
    _write_gradient_checks(checks, tables_dir / "gradient_check.md")

    with preserve_random_state():
        np.random.seed(seed)
        best_lambda, validation_accuracy, models_by_lambda = search_lambda(
            X_train,
            y_train,
            X_valid,
            y_valid,
            lambda_list=LAMBDA_LIST,
            random_state=seed,
            plot_path=figures_dir / "lambda_search_costs.png",
            input_layer_size=input_size,
            hidden_layer_sizes=hidden_layer_sizes,
            num_labels=num_labels,
            iterations=iterations,
        )
    plot_validation_error(
        LAMBDA_LIST,
        validation_accuracy,
        best_lambda,
        output_path=figures_dir / "validation_error_vs_lambda.png",
        show=False,
    )
    best_thetas = models_by_lambda[best_lambda]
    test_predictions = predict_model(X_test, best_thetas)
    test_accuracy = float(np.mean(test_predictions == np.asarray(y_test).ravel()) * 100)

    learning_result = run_learning_curve(
        file_path=data_path,
        validation_size=1000,
        min_train_size=100,
        num_points=6,
        random_state=seed,
        hidden_layer_sizes=hidden_layer_sizes,
        iterations=iterations,
        learning_rate=0.8,
        lambda_=best_lambda,
        output_path=figures_dir / "learning_curve.png",
        partition_curves_output_path=figures_dir / "learning_curve_by_train_size.png",
        show=False,
    )

    with preserve_random_state():
        np.random.seed(seed)
        thetas, _, _, snapshots = train_model(
            X_train,
            y_train,
            input_layer_size=input_size,
            hidden_layer_sizes=hidden_layer_sizes,
            num_labels=num_labels,
            iterations=iterations,
            lambda_=best_lambda,
            snapshot_every=max(1, iterations // 80),
        )
    create_activation_gif(
        snapshots,
        output_path=figures_dir / "activation_evolution.gif",
        input_shape=(20, 20) if input_size == 400 else None,
    )

    with preserve_random_state():
        np.random.seed(seed)
        run_error_cases_visualization(
            file_path=data_path,
            lambda_list=LAMBDA_LIST,
            hidden_layer_sizes=hidden_layer_sizes,
            iterations=iterations,
            output_path=figures_dir / "error_cases.png",
            show=False,
            random_state=seed,
        )

    with preserve_random_state():
        optimizer_results = compare_optimizers(
            X_train,
            y_train,
            X_test,
            y_test,
            iterations_list=optimizer_iterations,
            lambda_list=LAMBDA_LIST,
            random_state=seed,
            input_layer_size=input_size,
            hidden_layer_sizes=hidden_layer_sizes,
            num_labels=num_labels,
        )
    export_comparison_table(
        optimizer_results,
        tables_dir / "optimizer_comparison.csv",
        tables_dir / "optimizer_comparison.md",
    )

    config = {
        "data_path": data_path,
        "seed": seed,
        "input_size": input_size,
        "hidden_layer_sizes": hidden_layer_sizes,
        "num_labels": num_labels,
        "iterations": iterations,
    }
    _write_report(
        output_dir / "REPORT.md",
        config,
        checks,
        {
            "best_lambda": best_lambda,
            "validation_accuracy": validation_accuracy[np.argmax(validation_accuracy)],
        },
        learning_result,
        optimizer_results,
        test_accuracy,
    )
    plt.close("all")
    return output_dir


def main():
    args = parse_args()
    output_dir = generate_report(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
        iterations=args.iterations,
        optimizer_iterations=args.optimizer_iterations,
        hidden_layer_sizes=args.hidden_layer_sizes,
    )
    print(f"Relatório gerado em: {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
