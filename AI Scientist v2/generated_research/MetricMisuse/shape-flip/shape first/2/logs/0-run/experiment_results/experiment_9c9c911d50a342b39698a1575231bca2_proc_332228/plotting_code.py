import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths / load data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


best_dev_swa = {}
best_test_swa = {}

# ------------- per-dataset plots
for dset in experiment_data:
    log = experiment_data[dset]
    epochs = log.get("epochs", [])
    # Loss curves ----------------------------------------------------------
    try:
        if epochs and log["losses"].get("train"):
            plt.figure()
            plt.plot(epochs, log["losses"]["train"], label="train")
            if log["losses"].get("dev"):
                plt.plot(epochs, log["losses"]["dev"], label="dev")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset} – Loss Curve")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting loss for {dset}: {e}")
        plt.close()

    # SWA curves -----------------------------------------------------------
    try:
        if epochs and log["metrics"].get("train_SWA"):
            plt.figure()
            plt.plot(epochs, log["metrics"]["train_SWA"], label="train_SWA")
            if log["metrics"].get("dev_SWA"):
                plt.plot(epochs, log["metrics"]["dev_SWA"], label="dev_SWA")
                best_dev_swa[dset] = max(log["metrics"]["dev_SWA"])
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(f"{dset} – SWA Curve")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_swa_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting SWA for {dset}: {e}")
        plt.close()

    # Test metric bar plot -------------------------------------------------
    try:
        tmet = log.get("test_metrics", {})
        if tmet:
            if "SWA" in tmet:  # store for later comparison
                best_test_swa[dset] = tmet["SWA"]
            plt.figure()
            keys, vals = zip(*tmet.items())
            plt.bar(keys, vals, color="tab:blue")
            plt.ylim(0, 1)
            plt.title(f"{dset} – Test Metrics")
            for i, v in enumerate(vals):
                plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            fname = os.path.join(working_dir, f"{dset}_test_metrics.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting test metrics for {dset}: {e}")
        plt.close()

    # Confusion matrix -----------------------------------------------------
    try:
        yt = np.asarray(log.get("ground_truth", []))
        yp = np.asarray(log.get("predictions", []))
        if yt.size and yp.size:
            n_cls = int(max(yt.max(), yp.max()) + 1)
            cm = confusion_matrix(yt, yp, n_cls)
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dset} – Confusion Matrix")
            for i in range(n_cls):
                for j in range(n_cls):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dset}: {e}")
        plt.close()

# ------------- comparison plot across datasets ---------------------------
try:
    if len(best_dev_swa) > 1:
        plt.figure()
        names, vals = zip(*best_dev_swa.items())
        plt.bar(names, vals, color="tab:orange")
        plt.ylim(0, 1)
        plt.title("Comparison of Best Dev SWA Across Datasets")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.xticks(rotation=45, ha="right")
        fname = os.path.join(working_dir, "comparison_best_dev_SWA.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error plotting comparison SWA: {e}")
    plt.close()

print("Plotting complete – figures saved to", working_dir)
