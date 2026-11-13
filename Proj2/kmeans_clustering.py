"""Cluster a sample of face images with our K-means implementation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from utils import kmeans, pca, project_data


DATA_PATH = Path(__file__).parent / "data" / "face_data.mat"
OUTPUT_DIR = Path(__file__).parent / "results" / "kmeans"
MAX_IMAGES = 1000
K_VALUES = range(1, 9)
N_CLUSTERS = 4
IMAGES_PER_CLUSTER = 12
RANDOM_STATE = 42


def make_contact_sheet(images):
    ncols = 4
    nrows = int(np.ceil(IMAGES_PER_CLUSTER / ncols))
    image_size = 32
    gap = 3
    sheet = np.full(
        (nrows * image_size + (nrows - 1) * gap,
         ncols * image_size + (ncols - 1) * gap),
        np.max(images),
    )

    for index, image in enumerate(images[:IMAGES_PER_CLUSTER]):
        row, col = divmod(index, ncols)
        y = row * (image_size + gap)
        x = col * (image_size + gap)
        sheet[y : y + image_size, x : x + image_size] = image

    return sheet


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faces = loadmat(DATA_PATH)["X"]

    # A fixed sample keeps the hand-written K-means run practical and repeatable.
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(faces), size=min(MAX_IMAGES, len(faces)), replace=False)
    faces = faces[indices]

    inertias = []
    results = {}
    for k in K_VALUES:
        centroids, labels = kmeans(faces, k, num_iters=50, random_state=RANDOM_STATE)
        inertia = np.sum((faces - centroids[labels]) ** 2)
        inertias.append(inertia)
        results[k] = labels
        print(f"k={k}: inércia={inertia:.2f}")

    labels = results[N_CLUSTERS]
    centered_faces = faces - np.mean(faces, axis=0)
    U, _, _ = pca(centered_faces)
    projected_faces = project_data(centered_faces, U, 3)

    images = faces.reshape(-1, 32, 32).transpose(0, 2, 1)
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        2, 4, left=0.06, right=0.98, top=0.92, bottom=0.08,
        hspace=0.38, wspace=0.28, height_ratios=(1, 1.15),
    )

    for cluster in range(N_CLUSTERS):
        members = images[labels == cluster]
        print(f"Cluster {cluster + 1}: {len(members)} imagens")
        axis = fig.add_subplot(grid[0, cluster])
        axis.imshow(make_contact_sheet(members), cmap="gray")
        axis.set_title(f"Cluster {cluster + 1} (n={len(members)})")
        axis.axis("off")

    elbow_axis = fig.add_subplot(grid[1, :2])
    ks = list(K_VALUES)
    elbow_axis.plot(ks, np.array(inertias) / 1e9, marker="o", label="Inércia")
    chosen_inertia = inertias[ks.index(N_CLUSTERS)] / 1e9
    elbow_axis.axvline(
        N_CLUSTERS, color="tab:red", linestyle="--", alpha=0.75,
        label=f"k escolhido = {N_CLUSTERS}",
    )
    elbow_axis.scatter(
        [N_CLUSTERS], [chosen_inertia], color="tab:red", zorder=3,
    )
    elbow_axis.set_xticks(ks)
    elbow_axis.set_xlabel("Número de clusters (k)")
    elbow_axis.set_ylabel("Inércia (× 10⁹)")
    elbow_axis.set_title("Curva do cotovelo")
    elbow_axis.grid(True, alpha=0.25)
    elbow_axis.legend()

    axis = fig.add_subplot(grid[1, 2:], projection="3d")
    for cluster in range(N_CLUSTERS):
        members = labels == cluster
        axis.scatter(
            projected_faces[members, 0],
            projected_faces[members, 1],
            projected_faces[members, 2],
            label=f"Cluster {cluster + 1}",
            s=12,
        )
    axis.set(xlabel="PC 1", ylabel="PC 2", zlabel="PC 3")
    axis.set_title("Projeção PCA 3D das imagens")
    axis.legend(
        title="K-means nos 1.024 pixels",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=8,
        title_fontsize=8,
    )
    fig.text(0.02, 0.925, "A", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.475, "B", fontsize=18, fontweight="bold")

    panel_path = OUTPUT_DIR / "painel_clusters.png"
    fig.savefig(panel_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Painel salvo em: {panel_path}")


if __name__ == "__main__":
    main()
