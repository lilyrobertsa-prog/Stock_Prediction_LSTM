import json
import os

def save_results(results, ticker):
    os.makedirs("saved_results", exist_ok=True)

    filename = os.path.join(
        "saved_results",
        f"{ticker.upper()}_results.json"
    )

    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {filename}")