from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from utils import feature_normalize, pca, project_data, recover_data, show_N_images


DATA_PATH = Path(__file__).parent / "data" / "face_data.mat"
OUTPUT_DIR = Path(__file__).parent / "results" / "pca"

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
})


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faces_data = loadmat(DATA_PATH)
    faces = faces_data["X"]
    # The face vectors use MATLAB's column-major image layout.
    originals = faces[:100].reshape(-1, 32, 32).transpose(0, 2, 1)
    print(faces.shape)

    show_N_images(
        originals, 100, show=True,
        save_path=OUTPUT_DIR / "originals_100.png",
        title="100 faces originais",
    )


    X, mu, sigma = feature_normalize(faces)
    print("Dados normalizados")

    U, _, _ = pca(X)
    eigenfaces = U[:, :36].T.reshape(-1, 32, 32).transpose(0, 2, 1)
    show_N_images(
        eigenfaces, 36, show=True,
        save_path=OUTPUT_DIR / "eigenfaces_36.png",
        title="36 primeiras eigenfaces",
    )

    Z = project_data(X, U, 100)
    reconstructed = recover_data(Z, U, 100) * sigma + mu
    reconstructions = reconstructed[:100].reshape(-1, 32, 32).transpose(0, 2, 1)
    show_N_images(
        reconstructions, 100, show=True,
        save_path=OUTPUT_DIR / "reconstructions_100.png",
        title="100 reconstruções usando 100 PCs",
    )

    comparisons = np.concatenate((originals, reconstructions), axis=2)

    print(" original esquerda e reconstrução direita.")
    show_N_images(
        comparisons, 100, show=True,
        save_path=OUTPUT_DIR / "comparisons_100.png",
        title="Original à esquerda; reconstrução à direita",
    )
    save_summary_panel(originals, eigenfaces, reconstructions)


if __name__ == "__main__":
    main()
