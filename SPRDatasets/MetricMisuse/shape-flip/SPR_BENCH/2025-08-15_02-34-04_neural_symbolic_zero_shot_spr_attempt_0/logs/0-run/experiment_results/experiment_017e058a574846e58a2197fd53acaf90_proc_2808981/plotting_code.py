import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr is not None:
    # ------------------------------------------------------------------
    # print final test metrics
    test_swa, test_cwa, test_hwa = spr["metrics"]["test"]
    print(
        f"Test Metrics  ->  SWA={test_swa:.4f}  CWA={test_cwa:.4f}  HWA={test_hwa:.4f}"
    )

    # ------------------------------------------------------------------
    # Plot 1: Loss curves
    try:
        plt.figure()
        plt.plot(spr["losses"]["train"], label="Train")
        plt.plot(spr["losses"]["val"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Metric curves
    try:
        epochs = range(1, len(spr["metrics"]["train"]) + 1)
        train_metrics = np.array(spr["metrics"]["train"])  # shape [E, 3]
        val_metrics = np.array(spr["metrics"]["val"])
        labels = ["SWA", "CWA", "HWA"]
        plt.figure()
        for i, lab in enumerate(labels):
            plt.plot(epochs, train_metrics[:, i], label=f"Train-{lab}")
            plt.plot(epochs, val_metrics[:, i], label=f"Val-{lab}", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(
            "SPR_BENCH Accuracy Metrics Over Epochs\nSolid: Train, Dashed: Validation"
        )
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_metric_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric curve: {e}")
        plt.close()

    # ------------------------------------------------------------------
    # Plot 3: Ground truth vs Predictions distribution
    try:
        gt = np.array(spr["ground_truth"])
        pred = np.array(spr["predictions"])
        labels_sorted = sorted(list(set(gt) | set(pred)))
        gt_counts = [np.sum(gt == l) for l in labels_sorted]
        pred_counts = [np.sum(pred == l) for l in labels_sorted]

        x = np.arange(len(labels_sorted))
        width = 0.35
        plt.figure(figsize=(10, 4))
        plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
        plt.bar(x + width / 2, pred_counts, width, label="Predictions")
        plt.xticks(x, labels_sorted, rotation=45, ha="right")
        plt.ylabel("Count")
        plt.title("SPR_BENCH Label Distribution\nLeft: Ground Truth, Right: Predicted")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_label_distribution.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating label distribution plot: {e}")
        plt.close()
