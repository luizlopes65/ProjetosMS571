from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from utils import feature_normalize, pca, project_data, recover_data, show_N_images, save_summary_panel


DATA_PATH = Path(__file__).parent / "data" / "face_data.mat"
OUTPUT_DIR = Path(__file__).parent / "results" / "pca"

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
})

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faces_data = loadmat(DATA_PATH)
    faces = faces_data["X"]
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
