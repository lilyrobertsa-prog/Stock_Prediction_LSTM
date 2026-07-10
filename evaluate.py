import numpy as np
from sklearn.metrics import root_mean_squared_error

def evaluate_predictions(y_train_pred, y_test_pred, y_train_actual, y_test_actual):

    absolute_error = np.abs(y_test_actual - y_test_pred)

    y_train_pred = np.ravel(y_train_pred)
    y_test_pred = np.ravel(y_test_pred)
    y_train_actual = np.ravel(y_train_actual)
    y_test_actual = np.ravel(y_test_actual)


    train_rmse = root_mean_squared_error(
        y_train_actual,
        y_train_pred
    )

    test_rmse = root_mean_squared_error(
        y_test_actual,
        y_test_pred
    )

    direction_accuracy = np.mean(
        np.sign(y_test_pred) == np.sign(y_test_actual)
    )

    absolute_error = np.mean(np.abs(y_test_actual - y_test_pred))

    return train_rmse, test_rmse, direction_accuracy, absolute_error