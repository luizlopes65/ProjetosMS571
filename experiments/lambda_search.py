from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import get_data_partitioned, predict_model, train_model


LAMBDA_LIST = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]


def search_lambda(
    X_treino,
    y_treino,
    X_valid,
    y_valid,
    lambda_list=LAMBDA_LIST,
    **train_kwargs,
):
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid)
    validation_accuracy = []
    models_by_lambda = {}

    for lambda_valid in lambda_list:
        thetas_valid, _, _ = train_model(
            X_treino,
            y_treino,
            lambda_=lambda_valid,
            **train_kwargs,
        )
        models_by_lambda[lambda_valid] = thetas_valid
        pred_valid = predict_model(X_valid, thetas_valid)
        acc = np.mean(pred_valid == y_valid.ravel()) * 100
        validation_accuracy.append(acc)
        print(" Lambda:", lambda_valid, ":", acc, "%")

    best_index = np.argmax(validation_accuracy)
    return lambda_list[best_index], validation_accuracy, models_by_lambda


def plot_validation_error(
    lambda_list,
    validation_accuracy,
    best_lambda,
    output_path=None,
    show=True,
):
    """Gráfico do erro de validação (%) em função de λ (eixo x em escala log).

    λ = 0 não possui logaritmo, então é desenhado num offset fixo à esquerda
    de 0.001, marcado com "λ = 0". O λ escolhido é destacado em vermelho.
    """
    lambda_array = np.asarray(lambda_list, dtype=float)
    validation_error = np.asarray([100.0 - acc for acc in validation_accuracy])

    # posições no eixo log: troca λ=0 por um offset pequeno (não pode ser 0)
    x_plot = lambda_array.copy()
    zero_idx = None
    if np.any(lambda_array == 0):
        zero_idx = int(np.flatnonzero(lambda_array == 0)[0])
        x_plot[zero_idx] = 3e-4

    best_idx = int(np.argmax(validation_accuracy))
    best_error = validation_error[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(x_plot, validation_error, "o-", label="Erro de validação")

    if zero_idx is not None:
        ax.annotate(
            "λ = 0",
            (x_plot[zero_idx], validation_error[zero_idx]),
            textcoords="offset points",
            xytext=(-10, 10),
            fontsize=9,
        )

    ax.axvline(x_plot[best_idx], color="red", linestyle="--", alpha=0.7)
    ax.plot(
        x_plot[best_idx],
        best_error,
        "rs",
        markersize=8,
        label=f"λ* = {best_lambda} (erro {best_error:.2f}%)",
    )
    ax.set_xlabel("λ (escala logarítmica)")
    ax.set_ylabel("Erro de validação (%)")
    ax.set_title("Erro de validação em função de λ")
    ax.grid(True, which="both", alpha=0.3)
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


if __name__ == "__main__":
    X_treino, y_treino, X_valid, y_valid, X_teste, y_teste = get_data_partitioned()

    print("Validation Set Accuracy:")
    best_lambda, validation_accuracy, models_by_lambda = search_lambda(
        X_treino,
        y_treino,
        X_valid,
        y_valid,
    )

    thetas_best = models_by_lambda[best_lambda]
    pred_test = predict_model(X_teste, thetas_best)
    test_accuracy = np.mean(pred_test == np.asarray(y_teste).ravel()) * 100

    print("Best lambda:", best_lambda)
    print("Test Set Accuracy:", test_accuracy, "%")

    plot_validation_error(
        LAMBDA_LIST,
        validation_accuracy,
        best_lambda,
        output_path=PROJECT_ROOT / "visualizations" / "validation_error_vs_lambda.png",
    )
