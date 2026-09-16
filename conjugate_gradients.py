import numpy as np
from scipy.optimize import minimize

from train import get_data_partitioned, predict_model, train_model
from utils import computeCost, randInitializeWeights


ITERATIONS_LIST = (50, 100, 200, 400, 800)
LAMBDA_LIST = (0, 0.001, 0.01, 0.1, 1, 10)


def train_model_cg(
    X_treino,
    y_treino,
    input_layer_size=400,
    hidden_layer_size=25,
    num_labels=10,
    iterations=400,
    lambda_=1,
    initial_theta=None,
):
    X_treino = np.asarray(X_treino)
    y_treino = np.asarray(y_treino).reshape(-1, 1)

    if initial_theta is None:
        theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        initial_theta = np.concatenate((theta1.ravel(), theta2.ravel()))
    else:
        initial_theta = np.asarray(initial_theta, dtype=float).ravel()

    def objective(theta):
        _, _, _, cost, grad1, grad2 = computeCost(
            X_treino,
            y_treino,
            theta,
            input_layer_size,
            hidden_layer_size,
            num_labels,
            lambda_,
        )
        gradient = np.concatenate((grad1.ravel(), grad2.ravel()))
        return cost, gradient

    cost_history = [objective(initial_theta)[0]]
    result = minimize(
        objective,
        initial_theta,
        jac=True,
        method="CG",
        callback=lambda theta: cost_history.append(objective(theta)[0]),
        options={"maxiter": iterations},
    )

    theta1_end = (input_layer_size + 1) * hidden_layer_size
    theta1 = result.x[:theta1_end].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = result.x[theta1_end:].reshape(num_labels, hidden_layer_size + 1)
    return theta1, theta2, cost_history, result


def model_accuracy(X, y, theta1, theta2):
    predictions = predict_model(X, theta1, theta2)
    return np.mean(predictions == np.asarray(y).ravel()) * 100


def compare_optimizers(
    X_treino,
    y_treino,
    X_teste,
    y_teste,
    iterations_list=ITERATIONS_LIST,
    lambda_list=LAMBDA_LIST,
    random_state=42,
    input_layer_size=400,
    hidden_layer_size=25,
    num_labels=10,
):
    """Compara gradiente descendente e CG nas mesmas configurações."""
    X_treino = np.asarray(X_treino)
    y_treino = np.asarray(y_treino).reshape(-1, 1)
    results = []

    for iterations in iterations_list:
        for lambda_ in lambda_list:
            np.random.seed(random_state)
            theta1_gd, theta2_gd, _, _ = train_model(
                X_treino,
                y_treino,
                input_layer_size=input_layer_size,
                hidden_layer_size=hidden_layer_size,
                num_labels=num_labels,
                iterations=iterations,
                lambda_=lambda_,
            )
            accuracy_gd = model_accuracy(X_teste, y_teste, theta1_gd, theta2_gd)

            np.random.seed(random_state)
            theta1_cg, theta2_cg, _, _ = train_model_cg(
                X_treino,
                y_treino,
                input_layer_size=input_layer_size,
                hidden_layer_size=hidden_layer_size,
                num_labels=num_labels,
                iterations=iterations,
                lambda_=lambda_,
            )
            accuracy_cg = model_accuracy(X_teste, y_teste, theta1_cg, theta2_cg)

            results.append(
                {
                    "iterations": iterations,
                    "lambda": lambda_,
                    "gradient_descent_accuracy": accuracy_gd,
                    "conjugate_gradient_accuracy": accuracy_cg,
                }
            )

    return results


def print_comparison_table(results):
    print("\nComparison on test set")
    print(f"{'iterations':>10} {'lambda':>10} {'gradient descent (%)':>22} {'conjugate gradient (%)':>24}")
    for row in results:
        print(
            f"{row['iterations']:>10} {row['lambda']:>10.3g} "
            f"{row['gradient_descent_accuracy']:>22.2f} "
            f"{row['conjugate_gradient_accuracy']:>24.2f}"
        )

    best_gd = max(results, key=lambda row: row["gradient_descent_accuracy"])
    best_cg = max(results, key=lambda row: row["conjugate_gradient_accuracy"])
    print(
        f"\nBest GD: {best_gd['gradient_descent_accuracy']:.2f}% "
        f"(iterations={best_gd['iterations']}, lambda={best_gd['lambda']})"
    )
    print(
        f"Best CG: {best_cg['conjugate_gradient_accuracy']:.2f}% "
        f"(iterations={best_cg['iterations']}, lambda={best_cg['lambda']})"
    )


if __name__ == "__main__":
    X_treino, y_treino, _, _, X_teste, y_teste = get_data_partitioned()

    comparison = compare_optimizers(
        X_treino,
        y_treino,
        X_teste,
        y_teste,
    )
    print_comparison_table(comparison)
