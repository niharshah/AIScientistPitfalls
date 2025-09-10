import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths & data loading ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR", {})
if not spr:
    print("SPR data not found in experiment_data.npy")


# ---------- helper ----------
def safe_plot(fname, plot_fn):
    try:
        plot_fn()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating {fname}: {e}")
        plt.close()


# ---------- figure 1 : loss curves ----------
def plot_losses():
    tl = spr["losses"]["train"]
    vl = spr["losses"]["val"]
    ep = range(1, len(tl) + 1)
    plt.figure()
    plt.plot(ep, tl, label="Train Loss")
    plt.plot(ep, vl, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Dataset: Train vs Val Loss")
    plt.legend()


safe_plot("SPR_loss_curves.png", plot_losses)

# ---------- figure 2-4 : metric curves ----------
val_metrics = spr.get("metrics", {}).get("val", [])
cwa = [m["CWA"] for m in val_metrics]
swa = [m["SWA"] for m in val_metrics]
comp = [m["CompWA"] for m in val_metrics]
epochs = range(1, len(cwa) + 1)


def plot_metric(vals, ylabel, fname):
    def _p():
        plt.figure()
        plt.plot(epochs, vals, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"SPR Dataset: {ylabel} vs Epoch")

    safe_plot(fname, _p)


plot_metric(cwa, "Color-Weighted Accuracy", "SPR_CWA_curve.png")
plot_metric(swa, "Shape-Weighted Accuracy", "SPR_SWA_curve.png")
plot_metric(comp, "Complexity-Weighted Accuracy", "SPR_CompWA_curve.png")

# ---------- print final test metrics ----------
test_metrics = spr.get("metrics", {}).get("test", {})
if test_metrics:
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
