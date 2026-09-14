import numpy as np

from train import get_data_partitioned, train_model
from utils import prediction


LAMBDA_LIST = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]


def search_lambda(X_treino, y_treino, X_valid, y_valid, lambda_list=LAMBDA_LIST):
    """Select the lambda with the best validation-set accuracy."""
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid)
    validation_accuracy = []

    for lambda_valid in lambda_list:
        theta1_valid, theta2_valid, _, _ = train_model(
            X_treino,
            y_treino,
            lambda_=lambda_valid,
        )
        pred_valid = np.asarray(prediction(X_valid, theta1_valid, theta2_valid)).ravel()
        acc = np.mean(pred_valid == y_valid.ravel()) * 100
        validation_accuracy.append(acc)
        print(" Lambda:", lambda_valid, ":", acc, "%")

    best_index = np.argmax(validation_accuracy)
    return lambda_list[best_index], validation_accuracy


if __name__ == "__main__":
    X_treino, y_treino, X_valid, y_valid, _, _ = get_data_partitioned()

    print("Validation Set Accuracy:")
    best_lambda, _ = search_lambda(
        X_treino,
        y_treino,
        X_valid,
        y_valid,
    )
    print("Best lambda:", best_lambda)
