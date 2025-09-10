import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "SPR_BENCH"
if ds_key in experiment_data:
    data = experiment_data[ds_key]
    train_loss = data["losses"].get("train", [])
    val_loss = data["losses"].get("val", [])
    val_hwa = data["metrics"].get("val", [])
    preds = data.get("predictions", [])
    gts = data.get("ground_truth", [])
    n_epochs = max(len(train_loss), len(val_loss), len(val_hwa))

    # 1) Loss curve ---------------------------------------------------------
    try:
        plt.figure()
        if train_loss:
            plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train")
        if val_loss:
            plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) Validation HWA curve ----------------------------------------------
    try:
        plt.figure()
        if val_hwa:
            plt.plot(range(1, len(val_hwa) + 1), val_hwa, marker="o")
        plt.title("SPR_BENCH: Validation HWA over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic-Weighted Accuracy")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_hwa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curve: {e}")
        plt.close()

    # 3) Prediction vs Ground-Truth distribution ---------------------------
    try:
        if preds and gts:
            plt.figure()
            classes = sorted(set(gts + preds))
            gt_counts = [gts.count(c) for c in classes]
            pr_counts = [preds.count(c) for c in classes]
            x = np.arange(len(classes))
            width = 0.35
            plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
            plt.bar(x + width / 2, pr_counts, width, label="Predictions")
            plt.title("SPR_BENCH: Class Distribution (Test Set)")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.xticks(x, classes)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_class_dist.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating class distribution plot: {e}")
        plt.close()

    # Print final test HWA --------------------------------------------------
    test_hwa = data["metrics"].get("test", None)
    if test_hwa is not None:
        print(f"Final Test HWA: {test_hwa:.4f}")
