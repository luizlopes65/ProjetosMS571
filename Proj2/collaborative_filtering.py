"""Ajusta a filtragem colaborativa e imprime o ranking de filmes."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

from utils import cofi_cost_function, normalize_ratings

DATA_DIR = Path(__file__).parent / "data"
N_FEATURES = 100
REGULARIZATION = 0.0
SEED = 31
MAXITER = 1000
TOP_N = 10


def load_movie_names():
    names = []
    with open(DATA_DIR / "movie_names.txt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                names.append(parts[1])
    return names


def predict_ratings(Y, R, history=None, *, display=False):
    num_movies, num_users = Y.shape
    Y_norm, means = normalize_ratings(Y, R)
    rng = np.random.RandomState(SEED)
    X = rng.randn(num_movies, N_FEATURES) * 0.01
    theta = rng.randn(num_users, N_FEATURES) * 0.01
    initial = np.concatenate((X.ravel(), theta.ravel()))

    args = (Y_norm, R, num_users, num_movies, N_FEATURES, REGULARIZATION)

    def record_cost(params):
        history.append(cofi_cost_function(params, *args)[0])

    if history is not None:
        record_cost(initial)

    result = minimize(
        fun=cofi_cost_function,
        x0=initial,
        args=args,
        method="CG",
        jac=True,
        callback=record_cost if history is not None else None,
        options={"maxiter": MAXITER, "disp": display},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"O otimizador não convergiu: {result.message}")
    split = num_movies * N_FEATURES
    X = result.x[:split].reshape(num_movies, N_FEATURES)
    theta = result.x[split:].reshape(num_users, N_FEATURES)
    return X @ theta.T + means, result.nit, result.x.size


def load_movie_data():
    data = loadmat(DATA_DIR / "movie_data.mat")
    ratings, mask = data["Y"], data["R"]
    names = load_movie_names()
    if len(names) != ratings.shape[0]:
        raise ValueError(f"{len(names)} nomes para {ratings.shape[0]} filmes")
    return ratings, mask, names


def main():
    ratings, mask, names = load_movie_data()
    predictions, _, _ = predict_ratings(ratings, mask, display=True)
    scores = predictions.mean(axis=1)
    top_movies = np.argsort(scores)[-TOP_N:][::-1]
    for rank, index in enumerate(top_movies, start=1):
        print(f"{rank:2d}º - {names[index]} (Nota média: {scores[index]:.2f})")


if __name__ == "__main__":
    main()
