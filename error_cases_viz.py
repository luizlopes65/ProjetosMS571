import math

import matplotlib.pyplot as plt
import numpy as np

from lambda_search import LAMBDA_LIST, search_lambda
from train import get_data_partitioned, predict_model


def infer_image_shape(num_features):
    side = math.isqrt(num_features)
    if side * side != num_features:
        print("Erro de dimensão")
    return side, side


def plot_error_cases(
    X,
    y_true,
    y_pred,
    image_shape=None,
    max_images=25,
    title="Casos classificados incorretamente",
):
    """Mostra até ``max_images`` erros com rótulo verdadeiro e predito."""
    X = np.asarray(X)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    error_indices = np.flatnonzero(y_true != y_pred)[:max_images]

    if image_shape is None:
        image_shape = infer_image_shape(X.shape[1])

    if len(error_indices) == 0:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "Nenhum erro encontrado", ha="center", va="center")
        ax.axis("off")
        ax.set_title(title)
        return fig

    num_columns = min(5, len(error_indices))
    num_rows = math.ceil(len(error_indices) / num_columns)
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(2.2 * num_columns, 2.5 * num_rows),
        squeeze=False,
    )

    for axis, index in zip(axes.flat, error_indices):
        axis.imshow(X[index].reshape(image_shape), cmap="gray")
        axis.set_title(f"Verdadeiro: {y_true[index]}\nPredito: {y_pred[index]}")
        axis.axis("off")

    for axis in axes.flat[len(error_indices):]:
        axis.axis("off")

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    return fig


def run_error_cases_visualization(
    file_path="processed_data.csv",
    lambda_list=LAMBDA_LIST,
    hidden_layer_size=25,
    iterations=800,
    max_images=25,
    image_shape=None,
):
    X_treino, y_treino, X_valid, y_valid, X_teste, y_teste = get_data_partitioned(
        file_path=file_path
    )

    labels = np.concatenate(
        [
            np.asarray(y_treino).ravel(),
            np.asarray(y_valid).ravel(),
            np.asarray(y_teste).ravel(),
        ]
    )
    input_layer_size = np.asarray(X_treino).shape[1]
    num_labels = int(labels.max())

    best_lambda, validation_accuracy, models_by_lambda = search_lambda(
        X_treino,
        y_treino,
        X_valid,
        y_valid,
        lambda_list=lambda_list,
        input_layer_size=input_layer_size,
        hidden_layer_size=hidden_layer_size,
        num_labels=num_labels,
        iterations=iterations,
    )

    theta1, theta2 = models_by_lambda[best_lambda]
    y_true = np.asarray(y_teste).ravel()
    y_pred = predict_model(X_teste, theta1, theta2)
    accuracy = np.mean(y_true == y_pred) * 100
    best_validation_accuracy = validation_accuracy[np.argmax(validation_accuracy)]

    print(f"Lambda ótimo: {best_lambda}")
    print(f"Acurácia na validação: {best_validation_accuracy:.2f}%")
    print(f"Acurácia no teste: {accuracy:.2f}%")

    plot_error_cases(
        X_teste,
        y_true,
        y_pred,
        image_shape=image_shape,
        max_images=max_images,
        title=f"Erros | lambda={best_lambda} | acurácia={accuracy:.2f}%",
    )
    plt.show()

    return accuracy, best_lambda, y_true, y_pred


if __name__ == "__main__":
    run_error_cases_visualization()
