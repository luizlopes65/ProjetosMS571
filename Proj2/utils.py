import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def find_closest_centroids(X, centroids):
    squared_distances = np.sum(
        (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(squared_distances, axis=1)


def compute_centroids(X, assignments, k):
    centroids = np.empty((k, X.shape[1]), dtype=float)
    for centroid_index in range(k):
        points = X[assignments == centroid_index]
        if len(points) == 0:
            raise ValueError("Um centróide ficou sem exemplos atribuídos.")
        centroids[centroid_index] = np.mean(points, axis=0)
    return centroids


def initialize_centroids(X, k, random_state=None):
    if k > len(X):
        raise ValueError("k não pode ser maior que a quantidade de exemplos.")
    rng = np.random.default_rng(random_state)
    return X[rng.choice(len(X), size=k, replace=False)].astype(float, copy=True)


def kmeans(X, k, num_iters=100, random_state=None):
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
    normalized = np.zeros_like(X, dtype=float)
    np.divide(X - mu, sigma, out=normalized, where=sigma != 0)
    return normalized, mu, sigma


def pca(X):
    covariance = np.dot(X.T, X) / X.shape[0]
    return np.linalg.svd(covariance)


def project_data(X, U, k):
    return np.dot(X, U[:, :k])


def recover_data(Z, U, k):
    return np.dot(Z, U[:, :k].T)

def show_image(array, ind):
    plt.imshow(np.reshape(array[ind], (32, 32)).T, cmap="gray")
    plt.show()

def show_N_images(imgs, N, show=True, save_path=None, title=None):
    N = min(N, len(imgs))
    if N <= 0:
        raise ValueError("N precisa ser maior que zero e imgs não pode estar vazio.")

    ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))
    fig = plt.figure(figsize=(ncols, nrows))

    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(nrows, ncols),
        axes_pad=0.1,
        )

    axes = list(grid)
    for ax, im in zip(axes, imgs[:N]):
        ax.imshow(im, cmap="gray")
        ax.axis("off")

    for ax in axes[N:]:
        ax.set_visible(False)

    if title:
        fig.suptitle(title)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

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



def save_summary_panel(originals, eigenfaces, reconstructions):
    row_titles = [
        "Eigenfaces (PCs 1–9)",
        "Originais",
        "Reconstruções (100 PCs)",
    ]
    images_by_row = (eigenfaces[:9], originals[:9], reconstructions[:9])
    fig, axes = plt.subplots(3, 9, figsize=(20, 8))

    for row, images in enumerate(images_by_row):
        for col, image in enumerate(images):
            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"PC {col + 1}", fontsize=13)
            elif row == 1:
                axes[row, col].set_title(f"Exemplo {col + 1}", fontsize=13)

    fig.suptitle("PCA: eigenfaces e reconstruções", fontsize=22)
    fig.subplots_adjust(left=0.17, right=0.995, top=0.88, bottom=0.04, wspace=0.06, hspace=0.16)
    for row, title in enumerate(row_titles):
        position = axes[row, 0].get_position()
        fig.text(
            0.16, position.y0 + position.height / 2, title,
            ha="right", va="center", fontsize=14,
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for row, label in ((0, "A"), (1, "B")):
        title_box = axes[row, 0].title.get_window_extent(renderer)
        title_center_y = fig.transFigure.inverted().transform(title_box.get_points().mean(axis=0))[1]
        fig.text(
            0.16, title_center_y, label,
            ha="right", va="center", fontsize=24, fontweight="bold",
        )
    panel_path = OUTPUT_DIR / "pca_summary_9.png"
    fig.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Painel resumido salvo em: {panel_path}")

