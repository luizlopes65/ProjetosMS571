from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from preprocess_data import build_processed_dataset


class BuildProcessedDatasetTests(unittest.TestCase):
    def test_converts_decimal_commas_and_writes_headerless_dataset(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            images_path = directory / "images.csv"
            labels_path = directory / "labels.csv"
            output_path = directory / "processed.csv"
            pd.DataFrame([["0,5", "1"], ["2", "3,25"]]).to_csv(
                images_path, index=False, header=False
            )
            pd.DataFrame([1, 2]).to_csv(labels_path, index=False, header=False)

            dataset = build_processed_dataset(images_path, labels_path, output_path)

            self.assertEqual(dataset.shape, (2, 3))
            self.assertEqual(dataset.iloc[0].tolist(), [0.5, 1.0, 1.0])
            self.assertEqual(dataset.iloc[1].tolist(), [2.0, 3.25, 2.0])
            self.assertEqual(pd.read_csv(output_path, header=None).shape, (2, 3))

    def test_rejects_different_numbers_of_images_and_labels(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            images_path = directory / "images.csv"
            labels_path = directory / "labels.csv"
            pd.DataFrame([[1], [2]]).to_csv(images_path, index=False, header=False)
            pd.DataFrame([1]).to_csv(labels_path, index=False, header=False)

            with self.assertRaisesRegex(ValueError, "mesmo número"):
                build_processed_dataset(images_path, labels_path, directory / "out.csv")


if __name__ == "__main__":
    unittest.main()
