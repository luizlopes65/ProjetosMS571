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
PAIR_COLUMNS = (
    "indice_a",
    "classe_a",
    "indice_b",
    "classe_b",
    "similaridade_cosseno",
    "distancia_cosseno",
)
IEEE_TWO_COLUMN_WIDTH_IN = 7.16
PAIRS_PER_ROW = 2
PUBLISHED_PAIR_COUNT = 4


def infer_image_shape(num_features: int) -> tuple[int, int]:
    side = math.isqrt(num_features)
    if side * side != num_features:
        raise ValueError(
            "O número de atributos precisa formar uma imagem quadrada para a visualização."
        )
    return side, side


def format_class_label(value: object) -> str:
    """Formata rótulos inteiros sem a casa decimal introduzida pela CSV."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(numeric_value)) if numeric_value.is_integer() else str(value)


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
    pair_table: pd.DataFrame,
    image_shape: tuple[int, int] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Desenha os quatro pares persistidos mais similares em uma grade 2×2.

    ``pair_table`` é a fonte de verdade para classes, índices e métricas. Esta
    função somente recupera os pixels dos índices informados e não recalcula
    similaridades ou distâncias.
    """
    X = np.asarray(X)
    missing_columns = set(PAIR_COLUMNS) - set(pair_table.columns)
    if missing_columns:
        raise ValueError(f"Tabela de pares sem colunas: {sorted(missing_columns)}")
    if pair_table.empty:
        raise ValueError("Nenhum par foi fornecido para a visualização.")
    if image_shape is None:
        image_shape = infer_image_shape(X.shape[1])

    # A tabela continua completa para auditoria; a figura usa somente os quatro
    # primeiros registros, já ordenados pela similaridade na etapa original.
    pair_table = pair_table.head(PUBLISHED_PAIR_COUNT).copy()
    num_pairs = len(pair_table)
    num_rows = math.ceil(num_pairs / PAIRS_PER_ROW)
    # Largura padrão da área que ocupa duas colunas em IEEEtran. As imagens
    # ficam próximas de 0,8 in de lado e os cabeçalhos mantêm 7,6 pt.
    figure_height = max(3.55, 0.18 + 1.52 * num_rows)
    fig = plt.figure(
        figsize=(IEEE_TWO_COLUMN_WIDTH_IN, figure_height),
        facecolor="white",
    )
    pair_grid = fig.add_gridspec(
        num_rows,
        PAIRS_PER_ROW,
        left=0.025,
        right=0.975,
        bottom=0.055,
        top=0.93,
        wspace=0.06,
        hspace=0.26,
    )

    for pair_number, pair in pair_table.reset_index(drop=True).iterrows():
        row, column = divmod(pair_number, PAIRS_PER_ROW)
        pair_slot = pair_grid[row, column]
        image_grid = pair_slot.subgridspec(1, 2, wspace=0.035)
        first_index = int(pair["indice_a"])
        second_index = int(pair["indice_b"])

        for image_column, index in enumerate((first_index, second_index)):
            axis = fig.add_subplot(image_grid[0, image_column])
            # Os pixels foram serializados por coluna; a transposição restaura
            # a orientação visual original da imagem 20×20.
            axis.imshow(X[index].reshape(image_shape).T, cmap="gray")
            axis.axis("off")

        pair_bounds = pair_slot.get_position(fig)
        fig.text(
            (pair_bounds.x0 + pair_bounds.x1) / 2,
            pair_bounds.y1 + 0.012,
            "classe "
            f"{format_class_label(pair['classe_a'])} vs "
            f"{format_class_label(pair['classe_b'])}\n"
            f"cos={pair['similaridade_cosseno']:.4f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="normal",
            linespacing=0.9,
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_paths = [output_path]
        if output_path.suffix.lower() != ".pdf":
            output_paths.append(output_path.with_suffix(".pdf"))
        for path in output_paths:
            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            print(f"Figura salva em: {path}")
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
        result,
        output_path=figure_path,
        show=show,
    )
    return result


def regenerate_cross_class_pairs_figure(
    data_path: str | Path,
    pair_table_path: str | Path,
    figure_path: str | Path,
    show: bool = False,
) -> pd.DataFrame:
    """Gera a figura a partir da tabela persistida, sem nova busca de pares."""
    data = pd.read_csv(data_path, header=None)
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    pair_table = pd.read_csv(pair_table_path)

    missing_columns = set(PAIR_COLUMNS) - set(pair_table.columns)
    if missing_columns:
        raise ValueError(f"Tabela de pares sem colunas: {sorted(missing_columns)}")
    for pair in pair_table.itertuples(index=False):
        if not (0 <= pair.indice_a < len(X) and 0 <= pair.indice_b < len(X)):
            raise ValueError("A tabela contém índice fora dos limites dos dados.")
        if y[pair.indice_a] != pair.classe_a or y[pair.indice_b] != pair.classe_b:
            raise ValueError("As classes persistidas não correspondem aos índices informados.")

    plot_cross_class_pairs(X, pair_table, output_path=figure_path, show=show)
    return pair_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encontra pares interclasse com alta similaridade de cosseno."
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--table-path", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument(
        "--pairs-table",
        type=Path,
        help="Regenera a figura da tabela existente, sem recalcular os pares.",
    )
    args = parser.parse_args()
    if args.pairs_table is not None:
        regenerate_cross_class_pairs_figure(
            data_path=args.data_path,
            pair_table_path=args.pairs_table,
            figure_path=args.figure_path,
        )
    else:
        run_cosine_similarity_experiment(
            data_path=args.data_path,
            top_k=args.top_k,
            figure_path=args.figure_path,
            table_path=args.table_path,
        )
