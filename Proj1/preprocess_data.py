"""Prepara os CSVs brutos de imagens e rótulos para o treinamento.

Os arquivos de origem usam vírgula como separador decimal em parte dos valores.
O arquivo gerado contém uma linha por exemplo: 400 pixels normalizados seguidos
do rótulo, sem cabeçalho.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_PATH = PROJECT_ROOT / "data" / "images.csv"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "data" / "labels.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "processed_data.csv"


def _parse_numeric_frame(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Converte números que usam vírgula decimal para valores de ponto flutuante."""
    normalized = frame.astype("string").apply(
        lambda column: column.str.replace(",", ".", regex=False)
    )
    try:
        numeric = normalized.apply(pd.to_numeric, errors="raise")
    except ValueError as error:
        raise ValueError(f"{source} contém valores que não são numéricos.") from error

    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contém valores ausentes ou não finitos.")
    return numeric


def build_processed_dataset(
    images_path: str | Path = DEFAULT_IMAGES_PATH,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
) -> pd.DataFrame:
    """Lê os dados brutos, valida-os e grava o dataset usado pelos experimentos."""
    images_path = Path(images_path)
    labels_path = Path(labels_path)
    output_path = Path(output_path)

    if limit is not None and limit <= 0:
        raise ValueError("limit deve ser um inteiro positivo.")
    if not images_path.is_file():
        raise FileNotFoundError(f"Arquivo de imagens não encontrado: {images_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Arquivo de rótulos não encontrado: {labels_path}")

    images = _parse_numeric_frame(
        pd.read_csv(images_path, header=None, dtype=str), images_path
    )
    labels = _parse_numeric_frame(
        pd.read_csv(labels_path, header=None, dtype=str), labels_path
    )
    if labels.shape[1] != 1:
        raise ValueError("O arquivo de rótulos deve conter exatamente uma coluna.")
    if len(images) != len(labels):
        raise ValueError(
            "Imagens e rótulos devem ter o mesmo número de exemplos: "
            f"{len(images)} e {len(labels)}."
        )

    if limit is not None:
        images = images.iloc[:limit]
        labels = labels.iloc[:limit]

    label_values = labels.iloc[:, 0].to_numpy(dtype=float)
    if not np.equal(label_values, np.floor(label_values)).all():
        raise ValueError("Os rótulos devem ser valores inteiros.")

    dataset = images.copy()
    dataset[images.shape[1]] = label_values.astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, header=False)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte os CSVs brutos em processed_data.csv."
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        help="Processa somente os primeiros N exemplos; útil para testes rápidos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_processed_dataset(
        images_path=args.images,
        labels_path=args.labels,
        output_path=args.output,
        limit=args.limit,
    )
    print(
        f"Dataset processado salvo em: {args.output} "
        f"({len(dataset)} exemplos, {dataset.shape[1] - 1} atributos)."
    )


if __name__ == "__main__":
    main()
