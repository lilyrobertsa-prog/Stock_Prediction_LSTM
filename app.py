from flask import Flask, request, render_template
import os
from main import run_experiment



app = Flask(__name__, template_folder = 'templates')


print("Looking for templates in:", os.path.abspath(app.template_folder))
print("Files there:", os.listdir(app.template_folder))


from flask import Flask, render_template, request
from main import run_experiment

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    #error = None

    if request.method == "POST":
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

    return render_template("index.html", results=results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=555, debug=True)

