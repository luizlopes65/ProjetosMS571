"""Cluster a sample of face images with our K-means implementation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from utils import kmeans, pca, project_data, show_N_images


DATA_PATH = Path(__file__).parent / "data" / "face_data.mat"
OUTPUT_DIR = Path(__file__).parent / "results" / "kmeans"
MAX_IMAGES = 1000
K_VALUES = range(1, 9)
N_CLUSTERS = 4
IMAGES_PER_CLUSTER = 12
RANDOM_STATE = 42

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
})


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


def save_individual_visualizations(ks, inertias, projected_faces, labels, images):
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.plot(ks, np.array(inertias) / 1e9, marker="o", label="Inércia")
    axis.axvline(
        N_CLUSTERS, color="tab:red", linestyle="--",
        label=f"k escolhido = {N_CLUSTERS}",
    )
    axis.scatter(
        [N_CLUSTERS], [inertias[ks.index(N_CLUSTERS)] / 1e9],
        color="tab:red", zorder=3,
    )
    axis.set_xticks(ks)
    axis.set_xlabel("Número de clusters (k)")
    axis.set_ylabel("Inércia (× 10⁹)")
    axis.set_title("Curva do cotovelo")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(OUTPUT_DIR / "elbow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 7))
    axis = fig.add_subplot(111, projection="3d")
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
    axis.legend(title="K-means nos 1.024 pixels", fontsize=9, title_fontsize=9)
    fig.savefig(OUTPUT_DIR / "clusters_3d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for cluster in range(N_CLUSTERS):
        members = images[labels == cluster]
        show_N_images(
            members,
            IMAGES_PER_CLUSTER,
            show=False,
            save_path=OUTPUT_DIR / f"cluster_{cluster + 1}.png",
            title=f"Cluster {cluster + 1} ({len(members)} imagens)",
        )
        plt.close()


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
    save_individual_visualizations(ks, inertias, projected_faces, labels, images)
    plt.show()
    print(f"Painel salvo em: {panel_path}")


if __name__ == "__main__":
    main()
