"""Encontra imagens de classes distintas com alta similaridade de cosseno.

O experimento compara os vetores achatados e, em seguida, reconstrói os pares
em 20×20. Assim, evidencia que o cosseno nos pixels não representa a estrutura
espacial que uma CNN pode explorar.

Uso:
    .venv/bin/python experiments/cosine_similarity_cases.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_PATH = PROJECT_ROOT / "processed_data.csv"
DEFAULT_FIGURE_PATH = PROJECT_ROOT / "visualizations" / "cosine_cross_class_pairs.png"
DEFAULT_TABLE_PATH = PROJECT_ROOT / "visualizations" / "cosine_cross_class_pairs.csv"


def infer_image_shape(num_features: int) -> tuple[int, int]:
    side = math.isqrt(num_features)
    if side * side != num_features:
        raise ValueError(
            "O número de atributos precisa formar uma imagem quadrada para a visualização."
        )
    return side, side


def find_high_cosine_cross_class_pairs(
    X: np.ndarray,
    y: np.ndarray,
    top_k: int = 10,
    block_size: int = 512,
) -> list[tuple[int, int, float]]:
    """Retorna os ``top_k`` pares interclasse de maior similaridade cosseno.

    A matriz de similaridade é calculada por blocos para não exigir uma matriz
    quadrada inteira em memória. Cada par aparece apenas uma vez (índice i < j).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X deve ser 2D e ter o mesmo número de linhas que y.")
    if top_k < 1 or block_size < 1:
        raise ValueError("top_k e block_size devem ser positivos.")

    norms = np.linalg.norm(X, axis=1)
    valid = norms > 0
    if valid.sum() < 2:
        raise ValueError("São necessárias pelo menos duas imagens com norma não nula.")

    X_unit = np.zeros_like(X)
    X_unit[valid] = X[valid] / norms[valid, np.newaxis]
    all_indices = np.arange(len(X))
    candidates: list[tuple[int, int, float]] = []

    for start in range(0, len(X), block_size):
        stop = min(start + block_size, len(X))
        similarities = X_unit[start:stop] @ X_unit.T

        for local_index, i in enumerate(range(start, stop)):
            if not valid[i]:
                continue
            eligible = (
                (all_indices > i)
                & valid
                & (y != y[i])
            )
            candidate_indices = all_indices[eligible]
            if len(candidate_indices) == 0:
                continue

            candidate_scores = similarities[local_index, candidate_indices]
            count = min(top_k, len(candidate_indices))
            selected = np.argpartition(candidate_scores, -count)[-count:]
            for position in selected:
                candidates.append(
                    (
                        i,
                        int(candidate_indices[position]),
                        float(candidate_scores[position]),
                    )
                )

    candidates.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
    return candidates[:top_k]


def plot_cross_class_pairs(
    X: np.ndarray,
    y: np.ndarray,
    pairs: list[tuple[int, int, float]],
    image_shape: tuple[int, int] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Desenha os pares mais similares como imagens 2D, lado a lado."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if not pairs:
        raise ValueError("Nenhum par foi fornecido para a visualização.")
    if image_shape is None:
        image_shape = infer_image_shape(X.shape[1])

    fig, axes = plt.subplots(
        len(pairs), 2, figsize=(5.5, 2.7 * len(pairs)), squeeze=False
    )
    for row, (first_index, second_index, similarity) in enumerate(pairs):
        distance = 1 - similarity
        for axis, index, title in (
            (
                axes[row, 0],
                first_index,
                f"A: classe {y[first_index]} | índice {first_index}\n"
                f"sim.={similarity:.4f}; dist.={distance:.4f}",
            ),
            (
                axes[row, 1],
                second_index,
                f"B: classe {y[second_index]} | índice {second_index}",
            ),
        ):
            # Os pixels foram serializados por coluna; a transposição restaura
            # a orientação visual original da imagem 20×20.
            axis.imshow(X[index].reshape(image_shape).T, cmap="gray")
            axis.set_title(title)
            axis.axis("off")

    fig.suptitle(
        "Classes distintas com alta similaridade de cosseno\n"
        "O vetor é semelhante, mas o arranjo espacial 2D é diferente.",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Figura salva em: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def run_cosine_similarity_experiment(
    data_path: str | Path = DEFAULT_DATA_PATH,
    top_k: int = 10,
    figure_path: str | Path | None = DEFAULT_FIGURE_PATH,
    table_path: str | Path | None = DEFAULT_TABLE_PATH,
    show: bool = True,
) -> pd.DataFrame:
    """Executa a busca, salva a tabela de pares e gera a figura 2D."""
    data = pd.read_csv(data_path, header=None)
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    pairs = find_high_cosine_cross_class_pairs(X, y, top_k=top_k)

    result = pd.DataFrame(
        [
            {
                "indice_a": first_index,
                "classe_a": y[first_index],
                "indice_b": second_index,
                "classe_b": y[second_index],
                "similaridade_cosseno": similarity,
                "distancia_cosseno": 1 - similarity,
            }
            for first_index, second_index, similarity in pairs
        ]
    )
    if not np.all(result["classe_a"].to_numpy() != result["classe_b"].to_numpy()):
        raise RuntimeError("A busca retornou um par da mesma classe.")
    if not result["similaridade_cosseno"].is_monotonic_decreasing:
        raise RuntimeError("Os pares não estão ordenados por similaridade decrescente.")

    if table_path is not None:
        table_path = Path(table_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(table_path, index=False)
        print(f"Tabela salva em: {table_path}")

    plot_cross_class_pairs(
        X,
        y,
        pairs,
        output_path=figure_path,
        show=show,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encontra pares interclasse com alta similaridade de cosseno."
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    args = parser.parse_args()
    run_cosine_similarity_experiment(
        data_path=args.data_path,
        top_k=args.top_k,
        figure_path=args.figure_path,
        table_path=args.table_path,
    )
