from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from train import get_data_partitioned, predict_model, train_model
from utils import computeCost, pack_thetas, randInitializeAllWeights


class TrainingTests(unittest.TestCase):
    def test_partition_sizes_are_deterministic(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory) / "dataset.csv"
            data = pd.DataFrame(
                [[index, index + 1, 1 + index % 2] for index in range(20)]
            )
            data.to_csv(data_path, index=False, header=False)

            first_partition = get_data_partitioned(
                data_path, pct_valid=0.1, pct_teste=0.2, random_state=7
            )
            second_partition = get_data_partitioned(
                data_path, pct_valid=0.1, pct_teste=0.2, random_state=7
            )

            self.assertEqual([len(part) for part in first_partition], [14, 14, 2, 2, 4, 4])
            self.assertTrue(first_partition[0].equals(second_partition[0]))

    def test_training_returns_weights_and_predictions_with_expected_shapes(self) -> None:
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([[1], [2], [2], [1]])
        np.random.seed(3)

        thetas, history, initial_thetas = train_model(
            X, y, hidden_layer_sizes=[3], num_labels=2, iterations=3
        )

        self.assertEqual([theta.shape for theta in thetas], [(3, 3), (2, 4)])
        self.assertEqual(len(history), 3)
        self.assertEqual([theta.shape for theta in initial_thetas], [(3, 3), (2, 4)])
        self.assertEqual(predict_model(X, thetas).shape, (4,))

    def test_regularized_gradient_matches_numeric_gradient_for_small_network(self) -> None:
        X = np.array([[0.1, 0.3], [0.6, 0.2], [0.2, 0.9]])
        y = np.array([[1], [2], [1]])
        layer_sizes = [2, 2, 2]
        np.random.seed(5)
        theta = pack_thetas(randInitializeAllWeights(layer_sizes))
        analytic_gradient = pack_thetas(
            computeCost(X, y, theta, layer_sizes, num_labels=2, Lambda=0.3)[3]
        )
        epsilon = 1e-5
        numeric_gradient = np.zeros_like(theta)
        for index in range(len(theta)):
            plus = theta.copy()
            minus = theta.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numeric_gradient[index] = (
                computeCost(X, y, plus, layer_sizes, 2, 0.3)[2]
                - computeCost(X, y, minus, layer_sizes, 2, 0.3)[2]
            ) / (2 * epsilon)

        relative_difference = np.linalg.norm(analytic_gradient - numeric_gradient) / (
            np.linalg.norm(analytic_gradient) + np.linalg.norm(numeric_gradient)
        )
        self.assertLess(relative_difference, 1e-7)


if __name__ == "__main__":
    unittest.main()
