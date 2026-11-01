"""Treinamento e avaliação da rede neural para o dataset processado."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from utils import (
    gradientDescent,
    pack_thetas,
    prediction,
    randInitializeAllWeights,
    unpack_thetas,
)


PROCESSED_DATA_PATH = Path(__file__).resolve().with_name("processed_data.csv")
DEFAULT_HIDDEN_LAYER_SIZES = (25,)


def _read_dataset(file_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê o dataset sem cabeçalho e separa atributos e rótulo."""
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset processado não encontrado: {file_path}. "
            "Execute `uv run python preprocess_data.py` primeiro."
        )
    data = pd.read_csv(file_path, header=None)
    if data.shape[1] < 2:
        raise ValueError("O dataset deve conter ao menos um atributo e um rótulo.")
    if data.empty:
        raise ValueError("O dataset não pode estar vazio.")
    return data.iloc[:, :-1], data.iloc[:, -1:]


def get_data_partitioned(
    file_path: str | Path = PROCESSED_DATA_PATH,
    pct_valid: float = 0.04,
    pct_teste: float = 0.2,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Embaralha e separa os dados em treino, validação e teste.

    ``pct_valid`` e ``pct_teste`` são frações do conjunto total; o restante é
    destinado ao treino. A mesma ``random_state`` reproduz a mesma partição.
    """
    if not 0 <= pct_valid < 1 or not 0 < pct_teste < 1:
        raise ValueError("pct_valid e pct_teste devem ser frações entre 0 e 1.")
    if pct_valid + pct_teste >= 1:
        raise ValueError("pct_valid + pct_teste deve ser menor que 1.")

    X, y = _read_dataset(file_path)
    data = pd.concat([X, y], axis=1).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]

    total_examples = len(X)
    validation_examples = int(pct_valid * total_examples)
    test_examples = int(pct_teste * total_examples)
    training_examples = total_examples - validation_examples - test_examples
    if min(training_examples, validation_examples, test_examples) < 1:
        raise ValueError("Cada partição precisa conter pelo menos um exemplo.")

    validation_end = training_examples + validation_examples
    return (
        X.iloc[:training_examples],
        y.iloc[:training_examples],
        X.iloc[training_examples:validation_end],
        y.iloc[training_examples:validation_end],
        X.iloc[validation_end:],
        y.iloc[validation_end:],
    )


def predict_model(
    X: np.ndarray | pd.DataFrame, thetas: Sequence[np.ndarray]
) -> np.ndarray:
    """Retorna uma classe prevista para cada amostra de ``X``."""
    return np.asarray(prediction(np.asarray(X, dtype=float), thetas)).ravel()


def train_model(
    X_treino: np.ndarray | pd.DataFrame,
    y_treino: np.ndarray | pd.DataFrame,
    input_layer_size: int | None = None,
    hidden_layer_sizes: Sequence[int] | None = None,
    num_labels: int = 10,
    learning_rate: float = 0.8,
    iterations: int = 800,
    lambda_: float = 1,
    snapshot_every: int | None = None,
):
    """Treina uma rede sigmoidal e retorna pesos, custos e pesos iniciais."""
    X_treino = np.asarray(X_treino, dtype=float)
    y_treino = np.asarray(y_treino).reshape(-1, 1)
    if X_treino.ndim != 2 or X_treino.shape[0] == 0:
        raise ValueError("X_treino deve ser uma matriz não vazia.")
    if len(X_treino) != len(y_treino):
        raise ValueError("X_treino e y_treino devem ter o mesmo número de exemplos.")
    if not np.isfinite(X_treino).all():
        raise ValueError("X_treino contém valores não finitos.")
    if input_layer_size is None:
        input_layer_size = X_treino.shape[1]
    if input_layer_size != X_treino.shape[1]:
        raise ValueError(
            "input_layer_size deve coincidir com o número de colunas de X_treino."
        )
    if hidden_layer_sizes is None:
        hidden_layer_sizes = DEFAULT_HIDDEN_LAYER_SIZES
    if iterations <= 0 or learning_rate <= 0 or lambda_ < 0:
        raise ValueError(
            "iterations e learning_rate devem ser positivos; lambda_ não pode ser negativo."
        )
    if snapshot_every is not None and snapshot_every <= 0:
        raise ValueError("snapshot_every deve ser positivo.")

    layer_sizes = [input_layer_size, *hidden_layer_sizes, num_labels]
    initial_thetas = randInitializeAllWeights(layer_sizes)
    initial_theta = pack_thetas(initial_thetas)
    snapshots: list[tuple[int, list[np.ndarray]]] = []

    def save_snapshot(iteration: int, thetas: list[np.ndarray]) -> None:
        if snapshot_every is not None and (
            iteration == 1
            or iteration % snapshot_every == 0
            or iteration == iterations
        ):
            snapshots.append((iteration, [theta.copy() for theta in thetas]))

    theta, cost_history = gradientDescent(
        X_treino,
        y_treino,
        initial_theta,
        learning_rate,
        iterations,
        lambda_,
        layer_sizes,
        snapshot_callback=save_snapshot if snapshot_every is not None else None,
    )
    result = (unpack_thetas(theta, layer_sizes), cost_history, initial_thetas)
    return result + (snapshots,) if snapshot_every is not None else result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina a rede neural no dataset processado."
    )
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.8)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument(
        "--hidden-layer-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_HIDDEN_LAYER_SIZES),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data_partitioned(
        args.data_path
    )
    thetas, cost_history, _ = train_model(
        X_train,
        y_train,
        input_layer_size=X_train.shape[1],
        hidden_layer_sizes=args.hidden_layer_sizes,
        num_labels=int(
            max(
                y_train.max().iloc[0],
                y_valid.max().iloc[0],
                y_test.max().iloc[0],
            )
        ),
        learning_rate=args.learning_rate,
        iterations=args.iterations,
        lambda_=args.lambda_,
    )
    accuracy = np.mean(predict_model(X_test, thetas) == y_test.to_numpy().ravel()) * 100
    print(f"Custo final: {cost_history[-1]:.6f}")
    print(f"Acurácia no teste: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
