import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp_key = ("ConcatEmb_NoEarlyFusion", "SPR_BENCH")
try:
    exp = experiment_data[exp_key[0]][exp_key[1]]
    losses_tr = exp["losses"]["train"]
    losses_val = exp["losses"]["val"]
    val_metrics = exp["metrics"]["val"]
    test_metrics = exp["metrics"]["test"]
    preds = exp.get("predictions", [])
    gts = exp.get("ground_truth", [])
except Exception as e:
    print(f"Malformed experiment data: {e}")
    losses_tr = losses_val = val_metrics = []
    test_metrics = {}
    preds = gts = []

epochs = list(range(1, len(losses_tr) + 1))

# ---------- Plot 1: Loss curves ----------
try:
    plt.figure()
    plt.plot(epochs, losses_tr, label="Train Loss")
    plt.plot(epochs, losses_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH RGCN – Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- Plot 2: Validation weighted accuracies ----------
try:
    plt.figure()
    cwa = [m["cwa"] for m in val_metrics]
    swa = [m["swa"] for m in val_metrics]
    cpx = [m["cpxwa"] for m in val_metrics]
    plt.plot(epochs, cwa, label="Color Wtd Acc (CWA)")
    plt.plot(epochs, swa, label="Shape Wtd Acc (SWA)")
    plt.plot(epochs, cpx, label="Complexity Wtd Acc (CpxWA)")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR_BENCH – Validation Weighted Accuracies")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# ---------- Plot 3: Test metrics bar chart ----------
try:
    plt.figure()
    bars = ["CWA", "SWA", "CpxWA"]
    values = [
        test_metrics.get("cwa", 0),
        test_metrics.get("swa", 0),
        test_metrics.get("cpxwa", 0),
    ]
    plt.bar(bars, values, color=["tab:blue", "tab:orange", "tab:green"])
    plt.ylim(0, 1)
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.title("SPR_BENCH – Test Weighted Accuracies")
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar plot: {e}")
    plt.close()

print(f"Plots saved to {working_dir}")
