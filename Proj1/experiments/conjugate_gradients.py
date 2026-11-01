from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import get_data_partitioned, predict_model, train_model
from utils import computeCost, randInitializeAllWeights, pack_thetas, unpack_thetas


ITERATIONS_LIST = (50, 100, 200, 400, 800)
LAMBDA_LIST = (0, 0.001, 0.01, 0.1, 1, 10)


def train_model_cg(
    X_treino,
    y_treino,
    input_layer_size=400,
    hidden_layer_sizes=None,
    num_labels=10,
    iterations=400,
    lambda_=1,
    initial_theta=None,
):
    if hidden_layer_sizes is None:
        hidden_layer_sizes = [25]
    layer_sizes = [input_layer_size, *hidden_layer_sizes, num_labels]
    X_treino = np.asarray(X_treino)
    y_treino = np.asarray(y_treino).reshape(-1, 1)

    if initial_theta is None:
        initial_theta = pack_thetas(randInitializeAllWeights(layer_sizes))
    else:
        initial_theta = np.asarray(initial_theta, dtype=float).ravel()

    def objective(theta):
        _, _, cost, grads = computeCost(
            X_treino,
            y_treino,
            theta,
            layer_sizes,
            layer_sizes[-1],
            lambda_,
        )
        gradient = pack_thetas(grads)
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

    thetas = unpack_thetas(result.x, layer_sizes)
    return thetas, cost_history, result


def model_accuracy(X, y, thetas):
    predictions = predict_model(X, thetas)
    return np.mean(predictions == np.asarray(y).ravel()) * 100


def compare_optimizers(
    X_treino,
    y_treino,
    X_valid,
    y_valid,
    iterations_list=ITERATIONS_LIST,
    lambda_list=LAMBDA_LIST,
    random_state=42,
    input_layer_size=400,
    hidden_layer_sizes=None,
    num_labels=10,
):
    """Compara GD e CG no grid de hiperparâmetros usando só validação."""
    X_treino = np.asarray(X_treino)
    y_treino = np.asarray(y_treino).reshape(-1, 1)
    results = []

    for iterations in iterations_list:
        for lambda_ in lambda_list:
            np.random.seed(random_state)
            thetas_gd, _, _ = train_model(
                X_treino,
                y_treino,
                input_layer_size=input_layer_size,
                hidden_layer_sizes=hidden_layer_sizes,
                num_labels=num_labels,
                iterations=iterations,
                lambda_=lambda_,
            )
            validation_accuracy_gd = model_accuracy(X_valid, y_valid, thetas_gd)

            np.random.seed(random_state)
            thetas_cg, _, _ = train_model_cg(
                X_treino,
                y_treino,
                input_layer_size=input_layer_size,
                hidden_layer_sizes=hidden_layer_sizes,
                num_labels=num_labels,
                iterations=iterations,
                lambda_=lambda_,
            )
            validation_accuracy_cg = model_accuracy(X_valid, y_valid, thetas_cg)

            results.append(
                {
                    "iterations": iterations,
                    "lambda": lambda_,
                    "gradient_descent_validation_accuracy": validation_accuracy_gd,
                    "conjugate_gradient_validation_accuracy": validation_accuracy_cg,
                }
            )

    return results


def select_optimizer_configs(results):
    """Escolhe uma configuração por otimizador exclusivamente pela validação."""
    return {
        "gradient_descent": max(
            results, key=lambda row: row["gradient_descent_validation_accuracy"]
        ),
        "conjugate_gradient": max(
            results, key=lambda row: row["conjugate_gradient_validation_accuracy"]
        ),
    }


def evaluate_selected_optimizer_configs(
    X_treino,
    y_treino,
    X_teste,
    y_teste,
    selected_configs,
    random_state=42,
    input_layer_size=400,
    hidden_layer_sizes=None,
    num_labels=10,
):
    """Retreina os dois modelos selecionados e consulta o teste uma vez cada."""
    evaluations = {}
    gd_config = selected_configs["gradient_descent"]
    np.random.seed(random_state)
    gd_thetas, _, _ = train_model(
        X_treino,
        y_treino,
        input_layer_size=input_layer_size,
        hidden_layer_sizes=hidden_layer_sizes,
        num_labels=num_labels,
        iterations=gd_config["iterations"],
        lambda_=gd_config["lambda"],
    )
    evaluations["gradient_descent"] = {
        "iterations": gd_config["iterations"],
        "lambda": gd_config["lambda"],
        "validation_accuracy": gd_config["gradient_descent_validation_accuracy"],
        "test_accuracy": model_accuracy(X_teste, y_teste, gd_thetas),
    }

    cg_config = selected_configs["conjugate_gradient"]
    np.random.seed(random_state)
    cg_thetas, _, _ = train_model_cg(
        X_treino,
        y_treino,
        input_layer_size=input_layer_size,
        hidden_layer_sizes=hidden_layer_sizes,
        num_labels=num_labels,
        iterations=cg_config["iterations"],
        lambda_=cg_config["lambda"],
    )
    evaluations["conjugate_gradient"] = {
        "iterations": cg_config["iterations"],
        "lambda": cg_config["lambda"],
        "validation_accuracy": cg_config["conjugate_gradient_validation_accuracy"],
        "test_accuracy": model_accuracy(X_teste, y_teste, cg_thetas),
    }
    return evaluations


def export_comparison_table(results, csv_path, markdown_path):
    """Salva a comparação dos otimizadores em CSV e Markdown."""
    import csv

    csv_path = Path(csv_path)
    markdown_path = Path(markdown_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "iterations",
        "lambda",
        "gradient_descent_validation_accuracy",
        "conjugate_gradient_validation_accuracy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "| Iterações | Lambda | GD validação (%) | CG validação (%) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
            f"| {row['iterations']} | {row['lambda']:.3g} | "
            f"{row['gradient_descent_validation_accuracy']:.2f} | "
            f"{row['conjugate_gradient_validation_accuracy']:.2f} |"
        for row in results
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_comparison_table(results, selected_evaluations=None):
    print("\nComparison selected by validation")
    print(f"{'iterations':>10} {'lambda':>10} {'GD val. (%)':>12} {'CG val. (%)':>12}")
    for row in results:
        print(
            f"{row['iterations']:>10} {row['lambda']:>10.3g} "
            f"{row['gradient_descent_validation_accuracy']:>12.2f} "
            f"{row['conjugate_gradient_validation_accuracy']:>12.2f}"
        )

    selected = select_optimizer_configs(results)
    best_gd = selected["gradient_descent"]
    best_cg = selected["conjugate_gradient"]
    print(
        f"\nBest GD: validation {best_gd['gradient_descent_validation_accuracy']:.2f}% "
        f"(iterations={best_gd['iterations']}, lambda={best_gd['lambda']})"
    )
    print(
        f"Best CG: validation {best_cg['conjugate_gradient_validation_accuracy']:.2f}% "
        f"(iterations={best_cg['iterations']}, lambda={best_cg['lambda']})"
    )
    if selected_evaluations is not None:
        for name, evaluation in selected_evaluations.items():
            print(f"{name}: test {evaluation['test_accuracy']:.2f}%")


if __name__ == "__main__":
    X_treino, y_treino, X_valid, y_valid, X_teste, y_teste = get_data_partitioned()

    comparison = compare_optimizers(
        X_treino,
        y_treino,
        X_valid,
        y_valid,
    )
    selected = select_optimizer_configs(comparison)
    evaluation = evaluate_selected_optimizer_configs(
        X_treino, y_treino, X_teste, y_teste, selected
    )
    print_comparison_table(comparison, evaluation)
