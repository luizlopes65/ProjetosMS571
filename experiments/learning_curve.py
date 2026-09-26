from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lambda_search import LAMBDA_LIST, search_lambda
from train import predict_model, train_model

PROCESSED_DATA_PATH = PROJECT_ROOT / "processed_data.csv"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"


DEFAULT_VALIDATION_SIZE = 1_000
DEFAULT_MIN_TRAIN_SIZE = 100
DEFAULT_NUM_POINTS = 6

# Configuração da execução direta do script.
FILE_PATH = PROCESSED_DATA_PATH
VALIDATION_SIZE = DEFAULT_VALIDATION_SIZE
MIN_TRAIN_SIZE = DEFAULT_MIN_TRAIN_SIZE
NUM_POINTS = DEFAULT_NUM_POINTS
HIDDEN_LAYER_SIZES = [25]
ITERATIONS = 800
LEARNING_RATE = 0.8
LAMBDA_CANDIDATES = LAMBDA_LIST
OUTPUT_PATH = VISUALIZATIONS_DIR / "learning_curve.png"
PARTITION_CURVES_OUTPUT_PATH = VISUALIZATIONS_DIR / "learning_curve_by_train_size.png"


def split_train_and_validation(
    file_path: str | Path = PROCESSED_DATA_PATH,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv(file_path, header=None).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    if not 0 < validation_size < len(data):
        raise ValueError(
            "validation_size deve estar entre 1 e o número total de exemplos - 1."
        )

    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1:].to_numpy()
    split_index = len(data) - validation_size

    return X[:split_index], y[:split_index], X[split_index:], y[split_index:]


def make_train_sizes(
    max_train_size: int,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    num_points: int = DEFAULT_NUM_POINTS,
) -> np.ndarray:
    if not 0 < min_train_size <= max_train_size:
        raise ValueError("min_train_size deve estar entre 1 e max_train_size.")
    if num_points < 2:
        raise ValueError("num_points deve ser pelo menos 2.")

    return np.unique(
        np.linspace(min_train_size, max_train_size, num_points, dtype=int)
    )


def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accuracy = np.mean(np.asarray(y_true).ravel() == np.asarray(y_pred).ravel())
    return float(100 * (1 - accuracy))


def _train_with_seed(seed: int, *args, **kwargs):
    previous_state = np.random.get_state()
    np.random.seed(seed)
    try:
        return train_model(*args, **kwargs)
    finally:
        np.random.set_state(previous_state)


def plot_partition_learning_curves(
    train_sizes: np.ndarray,
    cost_histories: list[np.ndarray],
    lambda_: float,
) -> plt.Figure:
    """Cria uma grade com a evolução do custo para cada tamanho de treino."""
    num_columns = min(3, len(train_sizes))
    num_rows = int(np.ceil(len(train_sizes) / num_columns))
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(5 * num_columns, 3.6 * num_rows),
        squeeze=False,
    )

    for axis, size, history in zip(axes.flat, train_sizes, cost_histories):
        axis.plot(np.arange(1, len(history) + 1), history)
        axis.set_title(f"Treino com {size} exemplos")
        axis.set_xlabel("Iteração")
        axis.set_ylabel("Custo regularizado")
        axis.grid(alpha=0.3)

    for axis in axes.flat[len(train_sizes):]:
        axis.set_visible(False)

    fig.suptitle(f"Evolução do custo por tamanho de treino ($\\lambda$={lambda_})")
    fig.tight_layout()
    return fig


def run_learning_curve(
    file_path: str | Path = PROCESSED_DATA_PATH,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    train_sizes: Iterable[int] | None = None,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    num_points: int = DEFAULT_NUM_POINTS,
    random_state: int = 42,
    hidden_layer_sizes: list[int] | None = None,
    iterations: int = 800,
    learning_rate: float = 0.8,
    lambda_: float = 1.0,
    output_path: str | Path | None = None,
    partition_curves_output_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, np.ndarray | int | list[np.ndarray]]:

    if hidden_layer_sizes is None:
        hidden_layer_sizes = [25]

    X_pool, y_pool, X_valid, y_valid = split_train_and_validation(
        file_path=file_path,
        validation_size=validation_size,
        random_state=random_state,
    )

    if train_sizes is None:
        sizes = make_train_sizes(len(X_pool), min_train_size, num_points)
    else:
        sizes = np.unique(np.asarray(list(train_sizes), dtype=int))
        if len(sizes) == 0 or sizes[0] < 1 or sizes[-1] > len(X_pool):
            raise ValueError(
                "Cada tamanho de treino deve estar entre 1 e o tamanho do pool."
            )

    input_layer_size = X_pool.shape[1]
    num_labels = int(max(y_pool.max(), y_valid.max()))
    train_errors = []
    validation_errors = []
    cost_histories = []

    print("tamanho_treino | erro_treino (%) | erro_validacao (%)")
    for point_index, size in enumerate(sizes):
        X_train, y_train = X_pool[:size], y_pool[:size]
        thetas, cost_history, _ = _train_with_seed(
            random_state + point_index,
            X_train,
            y_train,
            input_layer_size=input_layer_size,
            hidden_layer_sizes=hidden_layer_sizes,
            num_labels=num_labels,
            learning_rate=learning_rate,
            iterations=iterations,
            lambda_=lambda_,
        )

        train_error = classification_error(
            y_train, predict_model(X_train, thetas)
        )
        validation_error = classification_error(
            y_valid, predict_model(X_valid, thetas)
        )
        train_errors.append(train_error)
        validation_errors.append(validation_error)
        cost_histories.append(np.asarray(cost_history))
        print(f"{size:15d} | {train_error:15.2f} | {validation_error:18.2f}")

    train_errors_array = np.asarray(train_errors)
    validation_errors_array = np.asarray(validation_errors)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_errors_array, "o-", label="Treino")
    ax.plot(sizes, validation_errors_array, "o-", label="Validação (fixa)")
    ax.set(
        xlabel="Número de exemplos de treino",
        ylabel="Erro de classificação (%)",
        title=(
            "Curva de aprendizado da rede regularizada "
            f"($\\lambda$={lambda_}, validação={validation_size})"
        ),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Gráfico salvo em: {output_path}")

    partition_fig = None
    if partition_curves_output_path is not None:
        partition_fig = plot_partition_learning_curves(sizes, cost_histories, lambda_)
        partition_curves_output_path = Path(partition_curves_output_path)
        partition_curves_output_path.parent.mkdir(parents=True, exist_ok=True)
        partition_fig.savefig(partition_curves_output_path, dpi=150)
        print(f"Gráfico salvo em: {partition_curves_output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
        if partition_fig is not None:
            plt.close(partition_fig)

    return {
        "train_sizes": sizes,
        "train_errors": train_errors_array,
        "validation_errors": validation_errors_array,
        "cost_histories": cost_histories,
        "validation_size": validation_size,
    }


def _search_lambda_with_seed(seed: int, *args, **kwargs):
    """Executa a busca de regularização sem alterar o gerador global."""
    previous_state = np.random.get_state()
    np.random.seed(seed)
    try:
        return search_lambda(*args, **kwargs)
    finally:
        np.random.set_state(previous_state)


def run_learning_curve_with_best_lambda(
    file_path: str | Path = PROCESSED_DATA_PATH,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    lambda_candidates: Iterable[float] = LAMBDA_LIST,
    train_sizes: Iterable[int] | None = None,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    num_points: int = DEFAULT_NUM_POINTS,
    random_state: int = 42,
    hidden_layer_sizes: list[int] | None = None,
    iterations: int = 800,
    learning_rate: float = 0.8,
    output_path: str | Path | None = None,
    partition_curves_output_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, np.ndarray | int | float | list[np.ndarray]]:
    """Seleciona o melhor ``lambda`` e gera uma única curva de aprendizado."""
    X_pool, y_pool, X_valid, y_valid = split_train_and_validation(
        file_path=file_path,
        validation_size=validation_size,
        random_state=random_state,
    )
    input_layer_size = X_pool.shape[1]
    num_labels = int(max(y_pool.max(), y_valid.max()))

    best_lambda, validation_accuracy, _ = _search_lambda_with_seed(
        random_state,
        X_pool,
        y_pool,
        X_valid,
        y_valid,
        lambda_list=list(lambda_candidates),
        input_layer_size=input_layer_size,
        hidden_layer_sizes=hidden_layer_sizes,
        num_labels=num_labels,
        iterations=iterations,
        learning_rate=learning_rate,
    )
    best_validation_accuracy = validation_accuracy[np.argmax(validation_accuracy)]
    print(
        f"\nLambda ótimo: {best_lambda} "
        f"(acurácia na validação: {best_validation_accuracy:.2f}%)"
    )

    result = run_learning_curve(
        file_path=file_path,
        validation_size=validation_size,
        train_sizes=train_sizes,
        min_train_size=min_train_size,
        num_points=num_points,
        random_state=random_state,
        hidden_layer_sizes=hidden_layer_sizes,
        iterations=iterations,
        learning_rate=learning_rate,
        lambda_=best_lambda,
        output_path=output_path,
        partition_curves_output_path=partition_curves_output_path,
        show=show,
    )
    result["best_lambda"] = float(best_lambda)
    result["best_validation_accuracy"] = float(best_validation_accuracy)
    return result


if __name__ == "__main__":
    run_learning_curve_with_best_lambda(
        file_path=FILE_PATH,
        validation_size=VALIDATION_SIZE,
        lambda_candidates=LAMBDA_CANDIDATES,
        min_train_size=MIN_TRAIN_SIZE,
        num_points=NUM_POINTS,
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        output_path=OUTPUT_PATH,
        partition_curves_output_path=PARTITION_CURVES_OUTPUT_PATH,
    )
