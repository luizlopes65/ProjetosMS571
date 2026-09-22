from pathlib import Path

from utils import (
    randInitializeAllWeights,
    unpack_thetas,
    pack_thetas,
    prediction,
    gradientDescent,
)
import numpy as np
import pandas as pd


PROCESSED_DATA_PATH = Path(__file__).resolve().with_name("processed_data.csv")


def get_data_partitioned(
    file_path=PROCESSED_DATA_PATH,
    pct_valid=0.04,
    pct_teste=0.2,
    random_state=42,
):
    data = pd.read_csv(file_path, header=None)
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    X = data.drop(columns=data.columns[-1])
    y = data.drop(columns=data.columns[:-1])

    N_TOTAL = X.shape[0]
    NUM_TREINO = int((1 - pct_teste) * N_TOTAL)
    NUM_VALID = int(pct_valid * N_TOTAL)

    X_treino, y_treino = X.iloc[:NUM_TREINO], y.iloc[:NUM_TREINO]
    X_valid, y_valid = X.iloc[N_TOTAL - NUM_VALID:], y.iloc[N_TOTAL - NUM_VALID:]
    X_teste, y_teste = X.iloc[NUM_TREINO:N_TOTAL - NUM_VALID], y.iloc[NUM_TREINO:N_TOTAL - NUM_VALID]

    return X_treino, y_treino, X_valid, y_valid, X_teste, y_teste


def predict_model(X, thetas):
    """Retorna uma classe prevista para cada amostra de ``X``."""
    predictions = prediction(np.asarray(X), thetas)
    return np.asarray(predictions).ravel()


def train_model(
    X_treino,
    y_treino,
    input_layer_size=400,
    hidden_layer_sizes=[25],
    num_labels=10,
    learning_rate=0.8,
    iterations=800,
    lambda_=1,
    snapshot_every=None,
):
    layer_sizes = [input_layer_size, *hidden_layer_sizes, num_labels]
    initial_thetas = randInitializeAllWeights(layer_sizes)
    initial_theta = pack_thetas(initial_thetas)
    snapshots = []

    def save_snapshot(iteration, thetas):
        if snapshot_every is not None and (
            iteration == 1
            or iteration % snapshot_every == 0
            or iteration == iterations
        ):
            snapshots.append((iteration, [t.copy() for t in thetas]))

    theta, J_history = gradientDescent(
        X_treino,
        y_treino,
        initial_theta,
        learning_rate,
        iterations,
        lambda_,
        layer_sizes,
        snapshot_callback=save_snapshot if snapshot_every is not None else None,
    )

    thetas = unpack_thetas(theta, layer_sizes)

    result = thetas, J_history, initial_thetas
    if snapshot_every is not None:
        return result + (snapshots,)
    return result


if __name__ == "__main__":
    X_treino, y_treino, X_valid, y_valid, X_teste, y_teste = get_data_partitioned()

    thetas, J_history, _ = train_model(X_treino, y_treino)

    X_teste = X_teste.to_numpy()
    y_teste = y_teste.to_numpy()

    pred = predict_model(X_teste, thetas)
    y_teste_flat = y_teste.ravel()

    print("Test Set Accuracy:", np.mean(pred == y_teste_flat) * 100, "%")
