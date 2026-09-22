from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import get_data_partitioned, predict_model, train_model


LAMBDA_LIST = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
DEFAULT_PLOT_PATH = PROJECT_ROOT / "visualizations" / "lambda_search_costs.png"


def classification_cost(X, y, theta1, theta2, num_labels):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    m = X.shape[0]

    X_with_bias = np.hstack((np.ones((m, 1)), X))
    hidden = 1 / (1 + np.exp(-(X_with_bias @ theta1.T)))
    hidden_with_bias = np.hstack((np.ones((m, 1)), hidden))
    probabilities = 1 / (1 + np.exp(-(hidden_with_bias @ theta2.T)))

    targets = (y[:, np.newaxis] == np.arange(1, num_labels + 1)).astype(float)
    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)
    loss = -targets * np.log(probabilities) - (1 - targets) * np.log(
        1 - probabilities
    )
    return float(np.mean(np.sum(loss, axis=1)))


def plot_lambda_costs(lambda_list, training_costs, validation_costs, output_path):
    lambda_values = np.asarray(lambda_list, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        r"Influência de $\lambda$ em $J_{treino}$ e $J_{cv}$",
        y=0.98,
    )
    ax.plot(
        lambda_values,
        training_costs,
        "o-",
        label=r"$J_{treino}$",
    )
    ax.plot(
        lambda_values,
        validation_costs,
        "o-",
        label=r"$J_{cv}$",
    )
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xticks(lambda_values)
    ax.set_xticklabels([f"{value:g}" for value in lambda_values])
    ax.set(
        xlabel=r"Parâmetro de regularização $\lambda$",
        ylabel="Entropia cruzada",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    return fig


def search_lambda(
    X_treino,
    y_treino,
    X_valid,
    y_valid,
    lambda_list=LAMBDA_LIST,
    random_state=42,
    plot_path=None,
    **train_kwargs,
):
    X_treino = np.asarray(X_treino)
    y_treino = np.asarray(y_treino)
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid)
    num_labels = int(train_kwargs.get("num_labels", 10))
    validation_accuracy = []
    training_costs = []
    validation_costs = []
    models_by_lambda = {}

    previous_random_state = np.random.get_state()
    initial_weights_state = None
    if random_state is not None:
        np.random.seed(random_state)
        initial_weights_state = np.random.get_state()

    try:
        for lambda_valid in lambda_list:
            if initial_weights_state is not None:
                np.random.set_state(initial_weights_state)

            theta1_valid, theta2_valid, _, _ = train_model(
                X_treino,
                y_treino,
                lambda_=lambda_valid,
                **train_kwargs,
            )
            models_by_lambda[lambda_valid] = (theta1_valid, theta2_valid)

            training_costs.append(
                classification_cost(
                    X_treino, y_treino, theta1_valid, theta2_valid, num_labels
                )
            )
            validation_costs.append(
                classification_cost(
                    X_valid, y_valid, theta1_valid, theta2_valid, num_labels
                )
            )
            pred_valid = predict_model(X_valid, theta1_valid, theta2_valid)
            acc = np.mean(pred_valid == y_valid.ravel()) * 100
            validation_accuracy.append(acc)
            print(
                f"Lambda: {lambda_valid:g} | "
                f"J_treino: {training_costs[-1]:.4f} | "
                f"J_cv: {validation_costs[-1]:.4f} | "
                f"acurácia de validação: {acc:.2f}%"
            )
    finally:
        np.random.set_state(previous_random_state)

    if plot_path is not None:
        figure = plot_lambda_costs(
            lambda_list, training_costs, validation_costs, plot_path
        )
        plt.close(figure)
        print(f"Gráfico salvo em: {plot_path}")

    best_index = np.argmax(validation_accuracy)
    return lambda_list[best_index], validation_accuracy, models_by_lambda


if __name__ == "__main__":
    X_treino, y_treino, X_valid, y_valid, X_teste, y_teste = get_data_partitioned()

    print("Validation Set Accuracy:")
    best_lambda, _, models_by_lambda = search_lambda(
        X_treino,
        y_treino,
        X_valid,
        y_valid,
        plot_path=DEFAULT_PLOT_PATH,
    )

    theta1_best, theta2_best = models_by_lambda[best_lambda]
    pred_test = predict_model(X_teste, theta1_best, theta2_best)
    test_accuracy = np.mean(pred_test == np.asarray(y_teste).ravel()) * 100

    print("Best lambda:", best_lambda)
    print("Test Set Accuracy:", test_accuracy, "%")
