import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data.get("unpacked_gru", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# -------------------------- 1) Loss curves --------------------------------- #
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for name, run in runs.items():
        axes[0].plot(run["losses"]["train"], label=name)
        axes[1].plot(run["losses"]["val"], label=name)
    axes[0].set_title("Left: Training Loss")
    axes[1].set_title("Right: Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle("SPR Dataset: Training vs Validation Loss Curves")
    fig.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_loss_curves_train_val.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# -------------------------- 2) Validation HWA curves ------------------------ #
try:
    plt.figure(figsize=(6, 4))
    for name, run in runs.items():
        hwa_vals = [m[2] for m in run["metrics"]["val"]]
        plt.plot(np.arange(1, len(hwa_vals) + 1), hwa_vals, label=name)
    plt.title("SPR Dataset: Validation HWA Curves")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_val_hwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# -------------------------- 3) Test HWA bar plot ---------------------------- #
try:
    names, hwas = [], []
    for name, run in runs.items():
        names.append(name.replace("epochs_", "e"))
        hwas.append(run["metrics"]["test"][2])
    plt.figure(figsize=(6, 4))
    plt.bar(names, hwas, color="slateblue")
    plt.title("SPR Dataset: Test HWA by Max Epochs")
    plt.xlabel("Run")
    plt.ylabel("Test HWA")
    for i, v in enumerate(hwas):
        plt.text(i, v + 0.005, f"{v:.2f}", ha="center", va="bottom")
    plt.savefig(os.path.join(working_dir, "spr_test_hwa_bars.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HWA bar plot: {e}")
    plt.close()

# -------------------------- 4) Confusion matrix ----------------------------- #
try:
    # pick best run by test HWA
    best_run_key = max(runs, key=lambda k: runs[k]["metrics"]["test"][2])
    best = runs[best_run_key]
    y_true = np.array(best["ground_truth"])
    y_pred = np.array(best["predictions"])
    labels = sorted(set(y_true) | set(y_pred))
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    lab2idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        cm[lab2idx[t], lab2idx[p]] += 1

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.colorbar(im, ax=ax)
    plt.title(f"SPR Dataset: Confusion Matrix (Best run {best_run_key})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_confusion_matrix_best.png"))
    plt.close(fig)
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
