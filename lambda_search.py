import numpy as np

from train import get_data_partitioned, predict_model, train_model


LAMBDA_LIST = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]


def search_lambda(
    X_treino,
    y_treino,
    X_valid,
    y_valid,
    lambda_list=LAMBDA_LIST,
    **train_kwargs,
):
    X_valid = np.asarray(X_valid)
    y_valid = np.asarray(y_valid)
    validation_accuracy = []
    models_by_lambda = {}

    for lambda_valid in lambda_list:
        theta1_valid, theta2_valid, _, _ = train_model(
            X_treino,
            y_treino,
            lambda_=lambda_valid,
            **train_kwargs,
        )
        models_by_lambda[lambda_valid] = (theta1_valid, theta2_valid)
        pred_valid = predict_model(X_valid, theta1_valid, theta2_valid)
        acc = np.mean(pred_valid == y_valid.ravel()) * 100
        validation_accuracy.append(acc)
        print(" Lambda:", lambda_valid, ":", acc, "%")

    best_index = np.argmax(validation_accuracy)
    return lambda_list[best_index], validation_accuracy, models_by_lambda


if __name__ == "__main__":
    X_treino, y_treino, X_valid, y_valid, X_teste, y_teste = get_data_partitioned()

    print("Validation Set Accuracy:")
    best_lambda, _, models_by_lambda = search_lambda(
        X_treino,
        y_treino,
        X_valid,
        y_valid,
    )

    theta1_best, theta2_best = models_by_lambda[best_lambda]
    pred_test = predict_model(X_teste, theta1_best, theta2_best)
    test_accuracy = np.mean(pred_test == np.asarray(y_teste).ravel()) * 100

    print("Best lambda:", best_lambda)
    print("Test Set Accuracy:", test_accuracy, "%")
