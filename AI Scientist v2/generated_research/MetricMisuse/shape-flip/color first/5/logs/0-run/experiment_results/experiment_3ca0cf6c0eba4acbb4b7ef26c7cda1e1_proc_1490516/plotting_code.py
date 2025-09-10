import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["num_epochs"]["SPR"]  # dict like {'epochs_5': hist, ...}
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}


# helper to get integer epoch budget
def run_key_to_int(k):
    return int(k.split("_")[-1]) if "_" in k else int(k)


# ---------- 1) loss curves ----------
try:
    plt.figure()
    for k, hist in runs.items():
        ep = hist["epochs"]
        plt.plot(ep, hist["losses"]["train"], label=f"train_{run_key_to_int(k)}")
        plt.plot(ep, hist["losses"]["val"], "--", label=f"val_{run_key_to_int(k)}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR dataset – Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- 2) accuracy curves ----------
try:
    plt.figure()
    for k, hist in runs.items():
        ep = hist["epochs"]
        plt.plot(ep, hist["metrics"]["train"], label=f"train_{run_key_to_int(k)}")
        plt.plot(ep, hist["metrics"]["val"], "--", label=f"val_{run_key_to_int(k)}")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title("SPR dataset – Training vs Validation CpxWA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# ---------- 3) test set performance ----------
try:
    plt.figure()
    budgets = sorted(runs.keys(), key=run_key_to_int)
    xs = [run_key_to_int(k) for k in budgets]
    cpxwa = [runs[k]["test_CpxWA"] for k in budgets]
    bars = plt.bar(xs, cpxwa, color="skyblue")
    for bar, k in zip(bars, budgets):
        tloss = runs[k]["test_loss"]
        ttime = runs[k]["train_time_s"]
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"loss={tloss:.2f}\ntime={ttime:.0f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xlabel("Epoch Budget")
    plt.ylabel("Test Complexity-Weighted Accuracy")
    plt.title("SPR dataset – Test Performance vs Epoch Budget")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_test_performance.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test performance plot: {e}")
    plt.close()
