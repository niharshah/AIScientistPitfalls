import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------
# Load stored experiment dictionary
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR_BENCH", {})
if not spr:
    print("SPR_BENCH data missing – nothing to plot.")
    exit()

train_losses = spr["losses"]["train"]
val_losses = spr["losses"]["val"]
val_metrics = spr["metrics"]["val"]
epochs = list(range(1, len(train_losses) + 1))


# helper: extract metric over epochs
def metric_list(key):
    return [m[key] for m in val_metrics]


# -------------------- plotting helpers -------------------
def safe_plot(fn):
    try:
        fn()
    except Exception as err:
        print(f"Plot error: {err}")
    finally:
        plt.close()


# -------------------- plots ------------------------------
# 1) Loss curves
def _plot_loss():
    plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    plt.title("SPR_BENCH Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))


safe_plot(_plot_loss)


# 2) Validation accuracy
def _plot_acc():
    plt.figure()
    plt.plot(epochs, metric_list("acc"), color="green")
    plt.title("SPR_BENCH Validation Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_accuracy.png"))


safe_plot(_plot_acc)


# 3) Color-weighted accuracy
def _plot_cwa():
    plt.figure()
    plt.plot(epochs, metric_list("cwa"), color="orange")
    plt.title("SPR_BENCH Color-Weighted Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CWA")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_cwa.png"))


safe_plot(_plot_cwa)


# 4) Shape-weighted accuracy
def _plot_swa():
    plt.figure()
    plt.plot(epochs, metric_list("swa"), color="purple")
    plt.title("SPR_BENCH Shape-Weighted Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa.png"))


safe_plot(_plot_swa)


# 5) Complexity-weighted accuracy
def _plot_compwa():
    plt.figure()
    plt.plot(epochs, metric_list("compwa"), color="red")
    plt.title("SPR_BENCH Complexity-Weighted Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("CompWA")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_compwa.png"))


safe_plot(_plot_compwa)


# 6) Confusion matrix at last epoch (one figure only)
def _plot_confusion():
    preds = spr.get("predictions", [])
    gts = spr.get("ground_truth", [])
    if not preds or not gts:
        print("No predictions found for confusion matrix.")
        return
    labels = sorted(list(set(gts + preds)))
    mat = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(gts, preds):
        mat[t, p] += 1
    plt.figure()
    plt.imshow(mat, cmap="Blues")
    plt.title("SPR_BENCH Confusion Matrix (Final Epoch)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, mat[i, j], ha="center", va="center", color="black")
    plt.colorbar()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))


safe_plot(_plot_confusion)

# -------------------- quick console summary ---------------
if val_metrics:
    last = val_metrics[-1]
    print(
        f"Final-epoch metrics: ACC={last['acc']:.3f}, "
        f"CWA={last['cwa']:.3f}, SWA={last['swa']:.3f}, "
        f"CompWA={last['compwa']:.3f}"
    )
