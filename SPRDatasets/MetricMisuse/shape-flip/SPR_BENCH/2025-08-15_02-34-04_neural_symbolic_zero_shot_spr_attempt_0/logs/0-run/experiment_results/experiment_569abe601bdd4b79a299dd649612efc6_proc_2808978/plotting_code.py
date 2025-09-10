import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr is not None:
    losses_tr = spr["losses"]["train"]
    losses_val = spr["losses"]["val"]
    metrics_tr = np.array(spr["metrics"]["train"])  # shape (E,3)
    metrics_val = np.array(spr["metrics"]["val"])
    test_metrics = spr["metrics"]["test"]  # (SWA,CWA,HWA)
    y_true = np.array(spr["ground_truth"])
    y_pred = np.array(spr["predictions"])

    # ---------- 1) loss curve ----------
    try:
        plt.figure()
        plt.plot(losses_tr, label="Train")
        plt.plot(losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- 2) metric curves ----------
    try:
        epochs = range(1, len(metrics_tr) + 1)
        labels = ["SWA", "CWA", "HWA"]
        plt.figure()
        for i, lab in enumerate(labels):
            plt.plot(epochs, metrics_tr[:, i], "--", label=f"Train {lab}")
            plt.plot(epochs, metrics_val[:, i], "-", label=f"Val {lab}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Weighted Accuracy Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_metric_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # ---------- 3) test metric bar chart ----------
    try:
        plt.figure()
        x = np.arange(3)
        plt.bar(x, test_metrics, color=["tab:blue", "tab:orange", "tab:green"])
        plt.xticks(x, ["SWA", "CWA", "HWA"])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Test Weighted Accuracies")
        for i, v in enumerate(test_metrics):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar chart: {e}")
        plt.close()

    # ---------- 4) confusion matrix ----------
    try:
        unique_labels = sorted(set(y_true) | set(y_pred))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[label_to_idx[t], label_to_idx[p]] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(unique_labels)), unique_labels, rotation=45, ha="right")
        plt.yticks(range(len(unique_labels)), unique_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=7,
                )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- console output ----------
    print(f"Test metrics (SWA, CWA, HWA): {test_metrics}")
