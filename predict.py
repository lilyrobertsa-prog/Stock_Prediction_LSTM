import torch
import numpy as np
from sklearn.metrics import root_mean_squared_error

def make_predictions(model, X_train, X_test, y_train, y_test, target_scaler, device):
    model.eval()

    X_train = X_train.to(device)
    X_test = X_test.to(device)

    with torch.no_grad():
        y_train_pred = model(X_train)
        y_test_pred = model(X_test)

    y_train_pred = target_scaler.inverse_transform(
        y_train_pred.cpu().numpy()
    )

    y_test_pred = target_scaler.inverse_transform(
        y_test_pred.cpu().numpy()
    )

    y_train_actual = target_scaler.inverse_transform(
        y_train.cpu().numpy()
    )

    y_test_actual = target_scaler.inverse_transform(
        y_test.cpu().numpy()
    )

    return y_train_pred, y_test_pred, y_train_actual, y_test_actual

