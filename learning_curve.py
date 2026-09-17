from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lambda_search import LAMBDA_LIST
from train import predict_model, train_model


DEFAULT_VALIDATION_SIZE = 1_000
DEFAULT_MIN_TRAIN_SIZE = 100
DEFAULT_NUM_POINTS = 6

# Configuração da execução direta do script.
FILE_PATH = "processed_data.csv"
VALIDATION_SIZE = DEFAULT_VALIDATION_SIZE
MIN_TRAIN_SIZE = DEFAULT_MIN_TRAIN_SIZE
NUM_POINTS = DEFAULT_NUM_POINTS
HIDDEN_LAYER_SIZE = 25
ITERATIONS = 800
LEARNING_RATE = 0.8
LAMBDA_VALUES = LAMBDA_LIST
OUTPUT_PATH = "visualizations/learning_curves_by_lambda.png"


def split_train_and_validation(
    file_path: str | Path = "processed_data.csv",
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


def run_learning_curve(
    file_path: str | Path = "processed_data.csv",
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    train_sizes: Iterable[int] | None = None,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    num_points: int = DEFAULT_NUM_POINTS,
    random_state: int = 42,
    hidden_layer_size: int = 25,
    iterations: int = 800,
    learning_rate: float = 0.8,
    lambda_: float = 1.0,
    output_path: str | Path | None = None,
    show: bool = True,
) -> dict[str, np.ndarray | int]:

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

    print("tamanho_treino | erro_treino (%) | erro_validacao (%)")
    for point_index, size in enumerate(sizes):
        X_train, y_train = X_pool[:size], y_pool[:size]
        theta1, theta2, _, _ = _train_with_seed(
            random_state + point_index,
            X_train,
            y_train,
            input_layer_size=input_layer_size,
            hidden_layer_size=hidden_layer_size,
            num_labels=num_labels,
            learning_rate=learning_rate,
            iterations=iterations,
            lambda_=lambda_,
        )

        train_error = classification_error(
            y_train, predict_model(X_train, theta1, theta2)
        )
        validation_error = classification_error(
            y_valid, predict_model(X_valid, theta1, theta2)
        )
        train_errors.append(train_error)
        validation_errors.append(validation_error)
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

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "train_sizes": sizes,
        "train_errors": train_errors_array,
        "validation_errors": validation_errors_array,
        "validation_size": validation_size,
    }


def run_learning_curves_by_lambda(
    lambda_values: Iterable[float] = LAMBDA_LIST,
    file_path: str | Path = "processed_data.csv",
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    train_sizes: Iterable[int] | None = None,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    num_points: int = DEFAULT_NUM_POINTS,
    random_state: int = 42,
    hidden_layer_size: int = 25,
    iterations: int = 800,
    learning_rate: float = 0.8,
    output_path: str | Path | None = None,
    show: bool = True,
) -> dict[float, dict[str, np.ndarray | int]]:
    """Gera uma curva de treino/validação para cada configuração de ``lambda``.

    Todas as configurações usam a mesma partição e os mesmos tamanhos de
    treino. Para cada tamanho, a inicialização também é a mesma entre valores
    de ``lambda``, deixando a regularização como única diferença experimental.
    """
    lambdas = np.unique(np.asarray(list(lambda_values), dtype=float))
    if len(lambdas) == 0:
        raise ValueError("Informe pelo menos uma configuração de lambda.")

    results: dict[float, dict[str, np.ndarray | int]] = {}
    for lambda_value in lambdas:
        print(f"\nConfiguração: lambda={lambda_value:g}")
        results[float(lambda_value)] = run_learning_curve(
            file_path=file_path,
            validation_size=validation_size,
            train_sizes=train_sizes,
            min_train_size=min_train_size,
            num_points=num_points,
            random_state=random_state,
            hidden_layer_size=hidden_layer_size,
            iterations=iterations,
            learning_rate=learning_rate,
            lambda_=float(lambda_value),
            show=False,
        )

    num_columns = min(3, len(lambdas))
    num_rows = int(np.ceil(len(lambdas) / num_columns))
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(5 * num_columns, 3.8 * num_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for axis, lambda_value in zip(axes.flat, lambdas):
        result = results[float(lambda_value)]
        sizes = np.asarray(result["train_sizes"])
        axis.plot(sizes, result["train_errors"], "o-", label="Treino")
        axis.plot(sizes, result["validation_errors"], "o-", label="Validação fixa")
        axis.set_title(f"$\\lambda$ = {lambda_value:g}")
        axis.grid(alpha=0.3)

    for axis in axes.flat[len(lambdas):]:
        axis.set_visible(False)

    axes[0, 0].legend()
    fig.supxlabel("Número de exemplos de treino")
    fig.supylabel("Erro de classificação (%)")
    fig.suptitle(
        f"Curvas de aprendizado por regularização (validação fixa: {validation_size})"
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"\nGráfico salvo em: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results


if __name__ == "__main__":
    run_learning_curves_by_lambda(
        lambda_values=LAMBDA_VALUES,
        file_path=FILE_PATH,
        validation_size=VALIDATION_SIZE,
        min_train_size=MIN_TRAIN_SIZE,
        num_points=NUM_POINTS,
        hidden_layer_size=HIDDEN_LAYER_SIZE,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        output_path=OUTPUT_PATH,
    )
