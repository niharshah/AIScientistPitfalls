import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

if data is not None:
    epochs = np.array(data["epochs"])
    train_loss = np.array(data["losses"]["train"])
    dev_loss = np.array(data["losses"]["dev"])
    train_pha = np.array(data["metrics"]["train_PHA"])
    dev_pha = np.array(data["metrics"]["dev_PHA"])
    gt = np.array(data["ground_truth"])
    pred = np.array(data["predictions"])
    test_metrics = data["test_metrics"]
    n_cls = int(max(gt.max(), pred.max()) + 1)

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, dev_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("spr_bench – Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) PHA curves
    try:
        plt.figure()
        plt.plot(epochs, train_pha, label="Train")
        plt.plot(epochs, dev_pha, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("PHA")
        plt.title("spr_bench – PHA Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_pha_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating PHA curve: {e}")
        plt.close()

    # 3) Test metric bars
    try:
        plt.figure()
        bars = ("SWA", "CWA", "PHA")
        values = [test_metrics["SWA"], test_metrics["CWA"], test_metrics["PHA"]]
        plt.bar(bars, values, color=["skyblue", "salmon", "lightgreen"])
        plt.ylim(0, 1)
        plt.title("spr_bench – Test Metrics")
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar chart: {e}")
        plt.close()

    # 4) Confusion matrix
    try:
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for g, p in zip(gt, pred):
            cm[g, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.title("spr_bench – Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # 5) Class distribution comparison
    try:
        plt.figure(figsize=(8, 4))
        cls = np.arange(n_cls)
        width = 0.35
        counts_gt = np.bincount(gt, minlength=n_cls)
        counts_pred = np.bincount(pred, minlength=n_cls)
        plt.bar(cls - width / 2, counts_gt, width, label="Ground Truth")
        plt.bar(cls + width / 2, counts_pred, width, label="Predictions")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("spr_bench – Class Distribution (Left: GT, Right: Pred)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_class_distribution.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating class distribution plot: {e}")
        plt.close()
