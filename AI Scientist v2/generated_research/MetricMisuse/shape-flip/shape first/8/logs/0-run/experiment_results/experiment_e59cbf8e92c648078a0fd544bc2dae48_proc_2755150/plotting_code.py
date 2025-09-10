import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("SWA_experiment", {})
train_loss = ed.get("losses", {}).get("train", [])
val_loss = ed.get("losses", {}).get("val", [])
train_swa = ed.get("metrics", {}).get("train_swa", [])
val_swa = ed.get("metrics", {}).get("val_swa", [])
preds = np.array(ed.get("predictions", []))
gts = np.array(ed.get("ground_truth", []))
test_swa = ed.get("test_swa", None)

# ---------- plot 1: training loss ----------
try:
    plt.figure()
    plt.plot(
        range(1, len(train_loss) + 1), train_loss, label="Train Loss", color="tab:blue"
    )
    plt.title("SPR_BENCH – Training Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_SWA_train_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# ---------- plot 2: SWA curves ----------
try:
    plt.figure()
    x = range(1, len(train_swa) + 1)
    plt.plot(x, train_swa, label="Train SWA", color="tab:green")
    plt.plot(x, val_swa, label="Val SWA", color="tab:orange")
    plt.title("SPR_BENCH – Shape-Weighted Accuracy (SWA) vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_SWA_train_val_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# ---------- plot 3: test metrics bar ----------
try:
    plt.figure()
    if preds.size and gts.size:
        acc = (preds == gts).mean()
        err = 1 - acc
        bars = ["Test SWA", "Accuracy", "Error"]
        vals = [test_swa if test_swa is not None else 0, acc, err]
        colors = ["tab:purple", "tab:blue", "tab:red"]
        plt.bar(bars, vals, color=colors)
        plt.ylim(0, 1)
        plt.title("SPR_BENCH – Final Test Metrics")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    fname = os.path.join(working_dir, "SPR_BENCH_SWA_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics plot: {e}")
    plt.close()

# ---------- print evaluation metrics ----------
if preds.size and gts.size:
    print(f"Final Test Accuracy: {(preds == gts).mean():.3f}")
if test_swa is not None:
    print(f"Final Test Shape-Weighted Accuracy (SWA): {test_swa:.3f}")
