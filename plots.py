import base64
from io import BytesIO

import matplotlib

# Prevents Matplotlib GUI errors when running through Flask.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def figure_to_base64(fig):
    """
    Convert a Matplotlib figure into a base64 string
    that can be displayed directly in HTML.
    """
    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=120
    )

    buffer.seek(0)
    image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    buffer.close()
    plt.close(fig)

    return image


def prediction_plot(y_test, linear_predictions, lstm_predictions):
    """Plot actual returns against both model predictions."""

    #y_test = np.asarray(y_test).reshape(-1)
    #linear_predictions = np.asarray(linear_predictions).reshape(-1)
    #lstm_predictions = np.asarray(lstm_predictions).reshape(-1)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(y_test, label="Actual return", linewidth=1.5)
    ax.plot(
        linear_predictions,
        label="Linear regression",
        linewidth=1,
        alpha=0.8
    )
    ax.plot(
        lstm_predictions,
        label="LSTM",
        linewidth=1,
        alpha=0.8
    )

    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.set_title("Actual vs Predicted Daily Returns")
    ax.set_xlabel("Test observation")
    ax.set_ylabel("Daily return")
    ax.legend()
    ax.grid(alpha=0.25)

    return figure_to_base64(fig)


def error_plot(y_test, linear_predictions, lstm_predictions):
    """Plot the absolute prediction error for each model."""

    y_test = np.asarray(y_test).reshape(-1)
    linear_predictions = np.asarray(linear_predictions).reshape(-1)
    lstm_predictions = np.asarray(lstm_predictions).reshape(-1)

    linear_errors = np.abs(y_test - linear_predictions)
    lstm_errors = np.abs(y_test - lstm_predictions)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        linear_errors,
        label="Linear regression error",
        linewidth=1
    )
    ax.plot(
        lstm_errors,
        label="LSTM error",
        linewidth=1
    )

    ax.set_title("Absolute Prediction Error Over Time")
    ax.set_xlabel("Test observation")
    ax.set_ylabel("Absolute error")
    ax.legend()
    ax.grid(alpha=0.25)

    return figure_to_base64(fig)


def metric_plot(
    linear_rmse,
    lstm_rmse,
    linear_mae,
    lstm_mae
):
    """Compare model error metrics."""

    model_names = ["Linear regression", "LSTM"]

    rmse_values = [linear_rmse, lstm_rmse]
    mae_values = [linear_mae, lstm_mae]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        rmse_values,
        width,
        label="RMSE"
    )

    ax.bar(
        x + width / 2,
        mae_values,
        width,
        label="MAE"
    )

    ax.set_title("Model Error Comparison")
    ax.set_ylabel("Error")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    return figure_to_base64(fig)