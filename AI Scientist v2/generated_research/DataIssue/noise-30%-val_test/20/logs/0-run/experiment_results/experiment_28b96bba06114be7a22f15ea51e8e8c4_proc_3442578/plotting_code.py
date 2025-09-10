import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dset = "SPR_BENCH"
budgets = experiment_data.get("num_epochs", {}).get(dset, {})
best_setting, best_val_f1 = None, -1

# -------------------- summary print -----------------------
print("\nBest val-F1 per epoch budget:")
for tag, res in budgets.items():
    val_f1 = max(res["metrics"]["val"])
    print(f"{tag:12s} : {val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1, best_setting = val_f1, tag

# -------------------- loss curves -------------------------
try:
    plt.figure()
    for tag, res in budgets.items():
        epochs = np.arange(1, len(res["losses"]["train"]) + 1)
        plt.plot(epochs, res["losses"]["train"], label=f"train_{tag}")
        plt.plot(epochs, res["losses"]["val"], label=f"val_{tag}", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss Curves across Epoch Budgets (SPR_BENCH)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------------- F1 curves ---------------------------
try:
    plt.figure()
    for tag, res in budgets.items():
        epochs = np.arange(1, len(res["metrics"]["train"]) + 1)
        plt.plot(epochs, res["metrics"]["train"], label=f"train_{tag}")
        plt.plot(epochs, res["metrics"]["val"], label=f"val_{tag}", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Macro-F1 Curves across Epoch Budgets (SPR_BENCH)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# -------------------- confusion matrix --------------------
try:
    if best_setting:
        preds = np.array(budgets[best_setting]["predictions"])
        gts = np.array(budgets[best_setting]["ground_truth"])
        n_cls = max(preds.max(), gts.max()) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"Confusion Matrix – {best_setting} (SPR_BENCH)")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8,
                )
        fname = os.path.join(working_dir, f"SPR_BENCH_conf_matrix_{best_setting}.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
