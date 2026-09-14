from utils import randInitializeWeights, prediction, gradientDescent
import numpy as np 
import pandas as pd


def get_data_partitioned(
    file_path="processed_data.csv",
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


def train_model(
    X_treino,
    y_treino,
    input_layer_size=400,
    hidden_layer_size=25,
    num_labels=10,
    learning_rate=0.8,
    iterations=800,
    lambda_=1,
    snapshot_every=None,
):
    initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)
    initial_theta = np.append(initial_theta1.flatten(), initial_theta2.flatten())
    snapshots = []

    def save_snapshot(iteration, theta1, theta2):
        if snapshot_every is not None and (
            iteration == 1
            or iteration % snapshot_every == 0
            or iteration == iterations
        ):
            snapshots.append((iteration, theta1.copy(), theta2.copy()))

    theta, J_history = gradientDescent(
        X_treino,
        y_treino,
        initial_theta,
        learning_rate,
        iterations,
        lambda_,
        input_layer_size,
        hidden_layer_size,
        num_labels,
        snapshot_callback=save_snapshot if snapshot_every is not None else None,
    )

    theta1_end = (input_layer_size + 1) * hidden_layer_size
    theta1 = theta[:theta1_end].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = theta[theta1_end:].reshape(num_labels, hidden_layer_size + 1)

    result = theta1, theta2, J_history, (initial_theta1, initial_theta2)
    if snapshot_every is not None:
        return result + (snapshots,)
    return result



if __name__ == "__main__":
    X_treino, y_treino, X_valid, y_valid, X_teste, y_teste = get_data_partitioned()

    theta1, theta2, J_history, _ = train_model(X_treino, y_treino)

    X_teste = X_teste.to_numpy()
    y_teste = y_teste.to_numpy()

    pred = prediction(X_teste,theta1,theta2)

    pred = np.asarray(pred).ravel()
    y_teste_flat = y_teste.ravel()

    print("Training Set Accuracy:", np.mean(pred == y_teste_flat) * 100, "%")
