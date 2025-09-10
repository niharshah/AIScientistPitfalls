import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data = experiment_data.get("SPR_BENCH", {})
    losses_tr = (
        np.array(data["losses"]["train"])
        if data["losses"]["train"]
        else np.empty((0, 2))
    )
    losses_val = (
        np.array(data["losses"]["val"]) if data["losses"]["val"] else np.empty((0, 2))
    )
    metrics_val = (
        np.array(data["metrics"]["val"]) if data["metrics"]["val"] else np.empty((0, 4))
    )
    preds = np.array(data.get("predictions", []))
    gts = np.array(data.get("ground_truth", []))

    # -------- Plot 1: loss curves --------
    try:
        plt.figure()
        if losses_tr.size:
            plt.plot(losses_tr[:, 0], losses_tr[:, 1], label="Train Loss")
        if losses_val.size:
            plt.plot(losses_val[:, 0], losses_val[:, 1], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Training vs. Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # -------- Plot 2: validation metrics --------
    try:
        plt.figure()
        if metrics_val.size:
            plt.plot(metrics_val[:, 0], metrics_val[:, 1], label="SWA")
            plt.plot(metrics_val[:, 0], metrics_val[:, 2], label="CWA")
            plt.plot(metrics_val[:, 0], metrics_val[:, 3], label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Validation Metrics (SWA, CWA, HWA)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metrics plot: {e}")
        plt.close()

    # -------- Plot 3: confusion matrix --------
    try:
        if preds.size and gts.size:
            num_classes = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("SPR_BENCH Confusion Matrix")
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # -------- Plot 4: class distribution bar chart --------
    try:
        if preds.size and gts.size:
            classes = np.arange(int(max(preds.max(), gts.max()) + 1))
            pred_counts = np.array([np.sum(preds == c) for c in classes])
            gt_counts = np.array([np.sum(gts == c) for c in classes])
            width = 0.35
            plt.figure()
            plt.bar(classes - width / 2, gt_counts, width=width, label="Ground Truth")
            plt.bar(classes + width / 2, pred_counts, width=width, label="Predictions")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title("SPR_BENCH Class Distribution")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_class_distribution.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating distribution plot: {e}")
        plt.close()

    # -------- Print final evaluation number --------
    if metrics_val.size:
        final_hwa = metrics_val[-1, 3]
        print(f"Final Harmonic Weighted Accuracy (HWA): {final_hwa:.4f}")
