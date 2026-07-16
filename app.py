from flask import Flask, request, render_template
import os
import numpy as np
from main import run_experiment

from plots import prediction_plot, error_plot, metric_plot

from database import get_experiments, initialise_database
import json
import pandas as pd


app = Flask(__name__, template_folder="templates")
initialise_database()


print("Running app from:", os.getcwd())
print("App file:", __file__)
print("Looking for templates in:", os.path.abspath(app.template_folder))
print("Files there:", os.listdir(app.template_folder))


@app.route("/history")
def history():
    ticker = request.args.get("ticker", "").strip().upper()
    model_type = request.args.get("model_type", "").strip()

    experiments = get_experiments(
        ticker=ticker or None,
        model_type=model_type or None,
    )

    return render_template(
        "history.html",
        experiments=experiments,
        selected_ticker=ticker,
        selected_model=model_type,
    )

@app.route("/", methods=["GET", "POST"])
def home():
    ticker = None
    error = None
    results = None
    prediction_graph = None
    error_graph = None
    metric_graph = None


    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()

        #load metrics        
        results_filename = os.path.join("saved_results", f"{ticker}_results.json")

        with open(results_filename) as f:
            results = json.load(f)

        #load prediction data
        predictions_filename = os.path.join("saved_predictions", f"{ticker}_predictions.csv")

        prediction_df = pd.read_csv(predictions_filename)

        y_test_raw = prediction_df["Actual"].to_numpy()
        y_test_pred_lin = prediction_df["Linear"].to_numpy()
        y_test_pred_lstm = prediction_df["LSTM"].to_numpy()

        prediction_graph = prediction_plot(
            y_test_raw,
            y_test_pred_lin,
            y_test_pred_lstm
        )

        error_graph = error_plot(
            y_test_raw,
            y_test_pred_lin,
            y_test_pred_lstm
        )

        metric_graph = metric_plot(
            results["linear_test_rmse"],
            results["lstm_test_rmse"],
            results["linear_test_mae"],
            results["lstm_test_mae"]
        )

    return render_template(
        "index.html",
        results=results,
        ticker=ticker,
        error=error,
        prediction_graph = prediction_graph,
        error_graph = error_graph,
        metric_graph = metric_graph
    )

if __name__ == '__main__':
    app.run(debug = True, port=5001)

