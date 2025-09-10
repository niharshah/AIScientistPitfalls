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


# ---------- helpers ----------------------------------------------------
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# collect final SWA per dataset for later comparison
final_swa = {}

# ---------------- plotting per dataset ---------------------------------
for dset, log in experiment_data.items():
    epochs = log.get("epochs", [])
    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, log["losses"]["train"], label="Train")
        plt.plot(epochs, log["losses"]["dev"], label="Dev")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dset} Loss Curve")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dset}: {e}")
        plt.close()

    # 2) SWA curves
    try:
        plt.figure()
        plt.plot(epochs, log["metrics"]["train_SWA"], label="Train_SWA")
        plt.plot(epochs, log["metrics"]["dev_SWA"], label="Dev_SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title(f"{dset} SWA Curve")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_swa_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SWA curve for {dset}: {e}")
        plt.close()

    # 3) Test metric bar plot
    try:
        test_m = log.get("test_metrics", {})
        if test_m:
            plt.figure()
            keys, vals = zip(*test_m.items())
            plt.bar(keys, vals, color="tab:blue")
            plt.ylim(0, 1)
            plt.ylabel("Score")
            plt.title(f"{dset} Test Metrics")
            for i, v in enumerate(vals):
                plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
            fname = os.path.join(working_dir, f"{dset}_test_metrics.png")
            plt.savefig(fname)
            final_swa[dset] = test_m.get("SWA", np.nan)
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar plot for {dset}: {e}")
        plt.close()

    # 4) Confusion matrix
    try:
        y_true = np.asarray(log.get("ground_truth", []))
        y_pred = np.asarray(log.get("predictions", []))
        if y_true.size and y_pred.size:
            n_classes = int(max(y_true.max(), y_pred.max()) + 1)
            cm = confusion_matrix(y_true, y_pred, n_classes)
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dset} Confusion Matrix")
            for i in range(n_classes):
                for j in range(n_classes):
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
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

# ------------- comparison plot across datasets -------------------------
try:
    if len(final_swa) > 1:
        plt.figure()
        ds, vals = zip(*final_swa.items())
        plt.bar(ds, vals, color="tab:green")
        plt.ylim(0, 1)
        plt.ylabel("Final Test SWA")
        plt.title("Dataset Comparison: Final Test SWA")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fname = os.path.join(working_dir, "comparison_final_test_SWA.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()

print("Plotting complete; figures saved to", working_dir)
