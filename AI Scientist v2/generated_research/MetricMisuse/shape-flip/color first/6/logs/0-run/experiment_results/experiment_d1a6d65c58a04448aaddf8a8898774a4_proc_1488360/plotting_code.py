import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# Load experiment data
# ------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    epochs = np.array(exp["epochs"])
    tr_loss = np.array(exp["losses"]["train"])
    val_loss = np.array(exp["losses"]["val"])
    val_compwa = np.array(exp["metrics"]["val"])
    y_true = np.array(exp["ground_truth"])
    y_pred = np.array(exp["predictions"])

    # --------------------------------------------------------
    # 1. Loss curves
    # --------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("spr_bench – Training vs Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # --------------------------------------------------------
    # 2. Validation CompWA curve
    # --------------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, val_compwa, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        plt.title("spr_bench – Validation Complexity-Weighted Accuracy")
        plt.grid(True, alpha=0.3)
        fname = os.path.join(working_dir, "spr_bench_compwa_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA curve: {e}")
        plt.close()

    # --------------------------------------------------------
    # 3. Confusion matrix heat-map (test set)
    # --------------------------------------------------------
    try:
        classes = sorted(set(np.concatenate([y_true, y_pred])))
        n_cls = len(classes)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(n_cls), classes)
        plt.yticks(range(n_cls), classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("spr_bench – Confusion Matrix (Test)")
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
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------------------------------------------------------
    # Print stored test metrics
    # --------------------------------------------------------
    test_metrics = exp["metrics"]["test"]
    print("Stored test metrics:", test_metrics)
