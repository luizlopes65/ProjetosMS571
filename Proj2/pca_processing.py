import numpy as np
from scipy.io import loadmat

from utils import feature_normalize, pca, project_data, recover_data, show_N_images


def main():
    faces_data = loadmat("data/face_data.mat")
    faces = faces_data["X"]
    originals = faces[:100].reshape(-1, 32, 32)
    print(faces.shape)

    show_N_images(originals, 100, show=True)


    X, mu, sigma = feature_normalize(faces)
    print("Dados normalizados")

    U, _, _ = pca(X)
    eigenfaces = U[:, :36].T.reshape(-1, 32, 32)
    show_N_images(eigenfaces, 36, show=True)

    Z = project_data(X, U, 100)
    reconstructed = recover_data(Z, U, 100) * sigma + mu
    reconstructions = reconstructed[:100].reshape(-1, 32, 32)

    comparisons = np.concatenate((originals, reconstructions), axis=2)

    print(" original esquerda e reconstrução direita.")
    show_N_images(comparisons, 100)


if __name__ == "__main__":
    main()
