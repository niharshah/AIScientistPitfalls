import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    results = experiment_data["embed_dim"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    results = []


# helper: collect data if it exists
def collect(metric_key):
    xs, ys = [], {}
    for r in results:
        dim = r["embed_dim"]
        ys[dim] = [m[metric_key] for m in r["metrics"]["val"]]
        xs = list(range(1, len(ys[dim]) + 1))
    return xs, ys


# ---------- plot 1: loss curves ----------
try:
    plt.figure()
    for r in results:
        dim = r["embed_dim"]
        plt.plot(r["losses"]["train"], label=f"train dim={dim}", linestyle="--")
        plt.plot(r["losses"]["val"], label=f"val dim={dim}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_loss_curves.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 2: accuracy curves ----------
try:
    xs, ys = collect("acc")
    plt.figure()
    for dim, vals in ys.items():
        plt.plot(xs, vals, label=f"dim={dim}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("SPR: Validation Accuracy over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_accuracy_curves.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- plot 3: final metrics bar chart ----------
try:
    metrics = ["acc", "cwa", "swa", "caa"]
    dims = [r["embed_dim"] for r in results]
    width = 0.18
    x = np.arange(len(dims))
    plt.figure()
    for i, m in enumerate(metrics):
        vals = [r["metrics"]["val"][-1][m] for r in results]
        plt.bar(x + i * width - width * 1.5, vals, width, label=m.upper())
    plt.xticks(x, [str(d) for d in dims])
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Score")
    plt.title("SPR: Final-Epoch Validation Metrics")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_final_metrics.png")
    plt.savefig(fname)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating final metrics plot: {e}")
    plt.close()
