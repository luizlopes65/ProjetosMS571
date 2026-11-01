"""Operações numéricas usadas pela rede neural multicamada.

Os nomes públicos seguem a nomenclatura original do trabalho para não quebrar
os notebooks e scripts já existentes no repositório.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from tqdm import tqdm


Array = np.ndarray
SnapshotCallback = Callable[[int, list[Array]], None]


def validate_layer_sizes(layer_sizes: Sequence[int]) -> tuple[int, ...]:
    """Valida a arquitetura e retorna seus tamanhos como uma tupla."""
    if layer_sizes is None:
        raise ValueError("layer_sizes não pode ser None.")

    sizes = tuple(layer_sizes)
    if len(sizes) < 2:
        raise ValueError("layer_sizes deve conter entrada e saída.")
    if any(
        isinstance(size, bool)
        or not isinstance(size, (int, np.integer))
        or size <= 0
        for size in sizes
    ):
        raise ValueError("Todos os tamanhos de camada devem ser inteiros positivos.")
    return sizes


def sigmoid(z: Array | float) -> Array:
    """Função sigmoide com proteção contra overflow numérico."""
    clipped = np.clip(np.asarray(z, dtype=float), -500, 500)
    return 1 / (1 + np.exp(-clipped))


def sigmoidGradient(z: Array | float) -> Array:
    """Derivada da sigmoide (nome mantido por compatibilidade)."""
    activation = sigmoid(z)
    return activation * (1 - activation)


def unpack_thetas(theta: Array, layer_sizes: Sequence[int]) -> list[Array]:
    """Reconstrói as matrizes de pesos de um vetor achatado."""
    layer_sizes = validate_layer_sizes(layer_sizes)
    theta = np.asarray(theta, dtype=float).ravel()
    expected_size = sum(
        output_size * (input_size + 1)
        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:])
    )
    if theta.size != expected_size:
        raise ValueError(
            f"theta possui {theta.size} valores, mas a arquitetura exige {expected_size}."
        )

    thetas: list[Array] = []
    offset = 0
    for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        size = output_size * (input_size + 1)
        thetas.append(theta[offset : offset + size].reshape(output_size, input_size + 1))
        offset += size
    return thetas


def pack_thetas(thetas: Sequence[Array]) -> Array:
    """Achata as matrizes de pesos em um único vetor."""
    if not thetas:
        raise ValueError("thetas deve conter pelo menos uma matriz de pesos.")
    return np.concatenate([np.asarray(theta, dtype=float).ravel() for theta in thetas])


def randInitializeWeights(input_size: int, output_size: int) -> Array:
    """Inicializa os pesos de uma transição pela estratégia de Xavier."""
    if input_size <= 0 or output_size <= 0:
        raise ValueError("Os tamanhos de entrada e saída devem ser positivos.")
    epsilon = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-epsilon, epsilon, size=(output_size, input_size + 1))


def randInitializeAllWeights(layer_sizes: Sequence[int]) -> list[Array]:
    """Inicializa todas as transições da arquitetura informada."""
    layer_sizes = validate_layer_sizes(layer_sizes)
    return [
        randInitializeWeights(input_size, output_size)
        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


def _validate_training_data(
    X: Array,
    y: Array,
    input_size: int,
    num_labels: int,
) -> tuple[Array, Array]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim != 2 or X.shape[1] != input_size or len(X) == 0:
        raise ValueError("X deve ter uma linha por exemplo e o número correto de atributos.")
    if len(y) != len(X):
        raise ValueError("X e y devem ter o mesmo número de exemplos.")
    if not np.isfinite(X).all():
        raise ValueError("X contém valores não finitos.")
    if not np.equal(y, np.floor(y)).all() or np.any((y < 1) | (y > num_labels)):
        raise ValueError(f"y deve conter classes inteiras entre 1 e {num_labels}.")
    return X, y.astype(int)


def computeCost(
    X: Array,
    y: Array,
    theta: Array,
    layer_sizes: Sequence[int],
    num_labels: int,
    Lambda: float,
) -> tuple[float, list[Array], float, list[Array]]:
    """Calcula custo e gradientes, com e sem regularização L2."""
    layer_sizes = validate_layer_sizes(layer_sizes)
    if num_labels != layer_sizes[-1]:
        raise ValueError("num_labels deve ser igual ao tamanho da camada de saída.")
    if Lambda < 0:
        raise ValueError("Lambda não pode ser negativo.")

    X, y = _validate_training_data(X, y, layer_sizes[0], num_labels)
    thetas = unpack_thetas(theta, layer_sizes)
    example_count = len(X)
    targets = (y[:, np.newaxis] == np.arange(1, num_labels + 1)).astype(float)

    activations = [np.hstack((np.ones((example_count, 1)), X))]
    for layer_index, layer_theta in enumerate(thetas):
        next_activation = sigmoid(activations[-1] @ layer_theta.T)
        if layer_index < len(thetas) - 1:
            next_activation = np.hstack((np.ones((example_count, 1)), next_activation))
        activations.append(next_activation)

    probabilities = np.clip(activations[-1], 1e-12, 1 - 1e-12)
    cost = float(
        -np.sum(
            targets * np.log(probabilities)
            + (1 - targets) * np.log(1 - probabilities)
        )
        / example_count
    )
    regularized_cost = cost + Lambda / (2 * example_count) * sum(
        np.sum(layer_theta[:, 1:] ** 2) for layer_theta in thetas
    )

    deltas: list[Array | None] = [None] * len(thetas)
    deltas[-1] = activations[-1] - targets
    for layer_index in range(len(thetas) - 2, -1, -1):
        hidden_activations = activations[layer_index + 1][:, 1:]
        deltas[layer_index] = (
            deltas[layer_index + 1] @ thetas[layer_index + 1][:, 1:]
        ) * hidden_activations * (1 - hidden_activations)

    gradients = [
        delta.T @ activation / example_count
        for delta, activation in zip(deltas, activations)
        if delta is not None
    ]
    regularized_gradients = [
        gradient
        + Lambda
        / example_count
        * np.hstack((np.zeros((layer_theta.shape[0], 1)), layer_theta[:, 1:]))
        for gradient, layer_theta in zip(gradients, thetas)
    ]
    return cost, gradients, float(regularized_cost), regularized_gradients


def gradientDescent(
    X: Array,
    y: Array,
    theta: Array,
    alpha: float,
    nbr_iter: int,
    Lambda: float,
    layer_sizes: Sequence[int],
    snapshot_callback: SnapshotCallback | None = None,
) -> tuple[Array, list[float]]:
    """Executa gradient descent em lote completo e registra o custo por época."""
    if alpha <= 0 or nbr_iter <= 0:
        raise ValueError("alpha e nbr_iter devem ser positivos.")
    layer_sizes = validate_layer_sizes(layer_sizes)
    thetas = unpack_thetas(theta, layer_sizes)
    cost_history: list[float] = []

    for iteration in tqdm(range(1, nbr_iter + 1), desc="Treinando", leave=False):
        current_theta = pack_thetas(thetas)
        _, _, cost, gradients = computeCost(
            X, y, current_theta, layer_sizes, layer_sizes[-1], Lambda
        )
        thetas = [
            layer_theta - alpha * gradient
            for layer_theta, gradient in zip(thetas, gradients)
        ]
        cost_history.append(cost)
        if snapshot_callback is not None:
            snapshot_callback(iteration, [layer_theta.copy() for layer_theta in thetas])

    return pack_thetas(thetas), cost_history


def prediction(X: Array, theta: Sequence[Array]) -> Array:
    """Executa a propagação para frente e devolve classes entre 1 e K."""
    if not theta:
        raise ValueError("theta deve conter pelo menos uma matriz de pesos.")
    X = np.asarray(X, dtype=float)
    first_layer = np.asarray(theta[0])
    if X.ndim != 2 or X.shape[1] + 1 != first_layer.shape[1]:
        raise ValueError("X é incompatível com os pesos da primeira camada.")

    activations = np.hstack((np.ones((len(X), 1)), X))
    for layer_index, layer_theta in enumerate(theta):
        activations = sigmoid(activations @ np.asarray(layer_theta).T)
        if layer_index < len(theta) - 1:
            activations = np.hstack((np.ones((len(X), 1)), activations))
    return np.argmax(activations, axis=1) + 1
