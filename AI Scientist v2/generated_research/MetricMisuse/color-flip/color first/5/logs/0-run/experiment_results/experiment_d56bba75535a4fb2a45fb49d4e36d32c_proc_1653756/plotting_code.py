import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

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
    experiment_data = None

# ---------- plotting ----------
if experiment_data is not None and "SPR_BENCH" in experiment_data:
    data = experiment_data["SPR_BENCH"]

    # 1. Loss curves ----------------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        epochs = np.arange(1, len(data["losses"]["train"]) + 1)
        plt.plot(epochs, data["losses"]["train"], label="train")
        plt.plot(epochs, data["losses"]["val"], linestyle="--", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH — Loss Curves\nLeft: train (solid)  Right: val (dashed)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 2. Validation accuracy --------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        accs = [m["acc"] for m in data["metrics"]["val"]]
        epochs = np.arange(1, len(accs) + 1)
        plt.plot(epochs, accs, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH — Validation Accuracy over Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating validation accuracy plot: {e}")
        plt.close()

    # 3. Test metrics bar ------------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        metric_names = ["acc", "cwa", "swa", "ccwa"]
        values = [data["metrics"]["test"].get(m, np.nan) for m in metric_names]
        plt.bar(metric_names, values, color="skyblue")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH — Test Metrics")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

    # 4. Silhouette per cluster -----------------------------------------------
    try:
        sil_dict = data.get("clusters", {}).get("silhouette", {})
        if sil_dict:
            clusters, sil_vals = zip(*sorted(sil_dict.items()))
            plt.figure(figsize=(6, 4))
            plt.bar(clusters, sil_vals, color="coral")
            plt.xlabel("Cluster ID")
            plt.ylabel("Mean Silhouette")
            plt.ylim(-1, 1)
            plt.title("SPR_BENCH — Cluster Silhouette Scores")
            fname = os.path.join(working_dir, "SPR_BENCH_cluster_silhouette.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating silhouette plot: {e}")
        plt.close()

    # 5. Confusion matrix ------------------------------------------------------
    try:
        y_true = np.array(data.get("ground_truth", []))
        y_pred = np.array(data.get("predictions", []))
        if y_true.size and y_pred.size:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(4, 4))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            classes = np.arange(cm.shape[0])
            plt.xticks(classes, classes)
            plt.yticks(classes, classes)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("SPR_BENCH — Confusion Matrix")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------- print test metrics ----------
    print("=== SPR_BENCH Test Metrics ===")
    for k, v in data["metrics"]["test"].items():
        print(f"{k.upper():5s}: {v:.3f}")
