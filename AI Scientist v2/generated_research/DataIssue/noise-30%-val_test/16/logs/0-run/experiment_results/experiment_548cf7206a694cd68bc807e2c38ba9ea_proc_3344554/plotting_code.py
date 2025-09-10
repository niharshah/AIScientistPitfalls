import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to extract metric lists safely
def extract(metric_key):
    tr = [d[metric_key] for d in ed["metrics"]["train"]]
    vl = [d[metric_key] for d in ed["metrics"]["val"]]
    return tr, vl


ed = experiment_data.get("NoLenFeatCounts", {}).get("SPR_BENCH", {})
epochs = ed.get("epochs", [])

plots_info = [
    (
        "loss",
        ed.get("losses", {}).get("train", []),
        ed.get("losses", {}).get("val", []),
    ),
    ("accuracy", *extract("acc")) if ed else None,
    ("MCC", *extract("MCC")) if ed else None,
    ("RMA", *extract("RMA")) if ed else None,
]
plots_info = [p for p in plots_info if p is not None][:5]  # at most 5 plots

for name, train_vals, val_vals in plots_info:
    try:
        plt.figure()
        plt.plot(epochs, train_vals, label="Train")
        plt.plot(epochs, val_vals, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(name.capitalize())
        plt.title(f"SPR_BENCH – {name.capitalize()} Curve (Train vs. Val)")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_{name}_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot {name}: {e}")
        plt.close()

# print final test metrics if present
try:
    test_metrics = ed.get("test_metrics", {})
    if test_metrics:
        print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})
except Exception as e:
    print(f"Error printing test metrics: {e}")
