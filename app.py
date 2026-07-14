from flask import Flask, request, render_template
import os
import numpy as np
from main import run_experiment

from plots import prediction_plot, error_plot, metric_plot



app = Flask(__name__, template_folder = 'templates')


print("Looking for templates in:", os.path.abspath(app.template_folder))
print("Files there:", os.listdir(app.template_folder))


from flask import Flask, render_template, request
from main import run_experiment

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    prediction_graph = None
    error_graph = None
    metric_graph = None
    error = None

    if request.method == "POST":
        try:
            
            ticker = request.form["ticker"]
            start_date = request.form["start_date"]
            end_date = request.form["end_date"]

            ##settings panel (javascript or CSS)
            seq_length = int(request.form["seq_length"])
            num_epochs = int(request.form["num_epochs"])
            hidden_dim = int(request.form["hidden_dim"])
            batch_size = int(request.form["batch_size"])
            learning_rate = float(request.form["learning_rate"])

            results = run_experiment(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                SEQ_LENGTH=seq_length,
                NUM_EPOCHS=num_epochs,
                HIDDEN_DIM=hidden_dim,
                BATCH_SIZE=batch_size,
                LEARNING_RATE=learning_rate
            )

            y_test_raw = np.asarray(
                results["y_test_raw"]
            ).reshape(-1)

            y_test_pred_lstm = np.asarray(
                results["y_test_pred_lstm"]
            ).reshape(-1)

            y_test_pred_lin = np.asarray(
                results["y_test_pred_lin"]
            ).reshape(-1)


            print("Mean actual:", np.mean(y_test_raw))
            print("Mean linear:", np.mean(y_test_pred_lin))
            print("Mean LSTM:", np.mean(y_test_pred_lstm))

            print("Std actual:", np.std(y_test_raw))
            print("Std linear:", np.std(y_test_pred_lin))
            print("Std LSTM:", np.std(y_test_pred_lstm))

            # Make all arrays the same length
            minimum_length = min(
                len(y_test_raw),
                len(y_test_pred_lin),
                len(y_test_pred_lstm)
            )

            y_test_raw = y_test_raw[-minimum_length:]
            y_test_pred_lin = y_test_pred_lin[-minimum_length:]
            y_test_pred_lstm = y_test_pred_lstm[-minimum_length:]

            print("Actual:", y_test_raw.shape)
            print("Linear:", y_test_pred_lin.shape)
            print("LSTM:", y_test_pred_lstm.shape)

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
                results["abs_error_lin"],
                results["abs_error_l"]
            )

            print(prediction_graph is None)
            print(error_graph is None)
            print(metric_graph is None)

        except ValueError as e:
            error = str(e)
            print("ValueError:", e)

        except Exception:
            error = ("Something went wrong while running the model.")
            print("Unexpected error:", repr(e))

    return render_template("index.html", results=results, prediction_graph=prediction_graph, error_graph = error_graph, metric_graph = metric_graph, error = error)

if __name__ == '__main__':
    app.run(debug = False)

