import numpy as np


def find_closest_centroids(X, centroids):
    """Atribui cada exemplo ao centróide de distância euclidiana mínima."""
    squared_distances = np.sum(
        (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(squared_distances, axis=1)


def compute_centroids(X, assignments, k):
    """Calcula os ``k`` centróides a partir das atribuições de cada exemplo."""
    centroids = np.empty((k, X.shape[1]), dtype=float)
    for centroid_index in range(k):
        points = X[assignments == centroid_index]
        if len(points) == 0:
            raise ValueError("Um centróide ficou sem exemplos atribuídos.")
        centroids[centroid_index] = np.mean(points, axis=0)
    return centroids


def initialize_centroids(X, k, random_state=None):
    """Seleciona ``k`` exemplos distintos de ``X`` como centróides iniciais."""
    if k > len(X):
        raise ValueError("k não pode ser maior que a quantidade de exemplos.")
    rng = np.random.default_rng(random_state)
    return X[rng.choice(len(X), size=k, replace=False)].astype(float, copy=True)


def kmeans(X, k, num_iters=100, random_state=None):
    """Executa K-Means e retorna centróides e atribuições de 0 a ``k - 1``."""
    centroids = initialize_centroids(X, k, random_state)
    for _ in range(num_iters):
        assignments = find_closest_centroids(X, centroids)
        updated_centroids = compute_centroids(X, assignments, k)
        if np.allclose(centroids, updated_centroids):
            centroids = updated_centroids
            break
        centroids = updated_centroids
    return centroids, find_closest_centroids(X, centroids)


def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma, mu, sigma


def pca(X):
    covariance = X.T @ X / X.shape[0]
    return np.linalg.svd(covariance)


def project_data(X, U, k):
    return X @ U[:, :k]


def recover_data(Z, U, k):
    return Z @ U[:, :k].T


def cofi_cost_function(params, Y, R, num_users, num_movies, num_features, regularization):

    split = num_movies * num_features
    X = params[:split].reshape(num_movies, num_features)
    theta = params[split:].reshape(num_users, num_features)

    error = X @ theta.T - Y
    masked_error = error * R
    cost = 0.5 * np.sum(masked_error**2)
    cost += regularization / 2 * (np.sum(X**2) + np.sum(theta**2))

    X_grad = masked_error @ theta + regularization * X
    theta_grad = masked_error.T @ X + regularization * theta
    gradient = np.concatenate((X_grad.ravel(), theta_grad.ravel()))
    return cost, gradient


def normalize_ratings(Y, R):
    rated_count = np.sum(R, axis=1, keepdims=True)
    Y_mean = np.divide(
        np.sum(Y * R, axis=1, keepdims=True),
        rated_count,
        out=np.zeros_like(Y[:, :1], dtype=float),
        where=rated_count != 0,
    )
    Y_norm = (Y - Y_mean) * R
    return Y_norm, Y_mean


def gradient_descent(
    initial_parameters,
    Y,
    R,
    num_users,
    num_movies,
    num_features,
    alpha,
    num_iters,
    regularization,
):
    params = initial_parameters.astype(float, copy=True)
    cost_history = []

    for _ in range(num_iters):
        cost, gradient = cofi_cost_function(
            params,
            Y,
            R,
            num_users,
            num_movies,
            num_features,
            regularization,
        )
        params -= alpha * gradient
        cost_history.append(cost)

    return params, cost_history
