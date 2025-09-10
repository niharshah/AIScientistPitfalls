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
    runs = experiment_data["weight_decay"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}


# Helper to extract per-epoch arrays -------------------------------------------------
def extract_series(run, key):  # key in {'train','val'}
    """Return list of values for losses[key] or metrics[key]"""
    if key == "dwa":
        return [v for _, v in run["metrics"]["val"]]
    if key == "train_loss":
        return [v for _, v in run["losses"]["train"]]
    if key == "val_loss":
        return [v for _, v in run["losses"]["val"]]


# ---------- Plot 1: loss curves -----------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    # two subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    for wd_key, run in runs.items():
        epochs = np.arange(1, len(run["losses"]["train"]) + 1)
        ax1.plot(epochs, extract_series(run, "train_loss"), label=wd_key)
        ax2.plot(epochs, extract_series(run, "val_loss"), label=wd_key)
    ax1.set_title("Train Loss")
    ax2.set_title("Validation Loss")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy")
        ax.legend()
    plt.suptitle("SPR_BENCH Loss Curves across weight_decay")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ---------- Plot 2: DWA curves ------------------------------------------------------
try:
    plt.figure()
    for wd_key, run in runs.items():
        epochs = np.arange(1, len(run["metrics"]["val"]) + 1)
        plt.plot(epochs, extract_series(run, "dwa"), label=wd_key)
    plt.xlabel("Epoch")
    plt.ylabel("Dual-Weighted-Accuracy")
    plt.title("SPR_BENCH Validation DWA over Epochs")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_DWA_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating DWA curves plot: {e}")
    plt.close()

# ---------- Plot 3: final DWA bar chart --------------------------------------------
try:
    final_dwa = {wd_key: extract_series(run, "dwa")[-1] for wd_key, run in runs.items()}
    plt.figure()
    plt.bar(
        range(len(final_dwa)),
        list(final_dwa.values()),
        tick_label=list(final_dwa.keys()),
    )
    plt.ylabel("Final Dual-Weighted-Accuracy")
    plt.title("SPR_BENCH Final DWA per weight_decay")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_DWA_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final DWA bar plot: {e}")
    plt.close()

# ---------- Plot 4: confusion scatter for best run ---------------------------------
try:
    # pick best by final DWA
    best_wd = max(runs, key=lambda k: extract_series(runs[k], "dwa")[-1])
    best_run = runs[best_wd]
    y_true = np.array(best_run["ground_truth"])
    y_pred = np.array(best_run["predictions"])
    plt.figure()
    plt.scatter(
        y_true + 0.05 * np.random.randn(len(y_true)),
        y_pred + 0.05 * np.random.randn(len(y_pred)),
        alpha=0.6,
        s=10,
    )
    max_cls = max(y_true.max(), y_pred.max())
    plt.plot([0, max_cls], [0, max_cls], "k--", linewidth=1)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title(f"SPR_BENCH Predictions vs Ground Truth (Best: {best_wd})")
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_scatter_best_{best_wd}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating scatter plot: {e}")
    plt.close()
