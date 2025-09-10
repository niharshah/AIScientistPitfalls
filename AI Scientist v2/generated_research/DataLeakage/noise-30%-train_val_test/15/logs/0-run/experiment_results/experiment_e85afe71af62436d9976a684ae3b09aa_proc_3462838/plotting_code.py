import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["epochs_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    epochs = exp["epochs"]
    train_loss = exp["losses"]["train"]
    val_loss = exp["losses"]["val"]
    train_f1 = exp["metrics"]["train"]
    val_f1 = exp["metrics"]["val"]
    lrs = exp["lrs"]
    preds = np.array(exp["predictions"])
    gts = np.array(exp["ground_truth"])
    n_labels = len(set(np.concatenate([preds, gts]))) if len(preds) else 0

    # 1. Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2. Macro-F1 curves
    try:
        plt.figure()
        plt.plot(epochs, train_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_macroF1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # 3. Learning-rate schedule
    try:
        plt.figure()
        plt.plot(epochs, lrs, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("SPR_BENCH Learning-Rate Schedule")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_lr_schedule.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating LR plot: {e}")
        plt.close()

    # 4. Label distribution: ground-truth vs. predictions
    if n_labels:
        try:
            gt_counts = np.bincount(gts, minlength=n_labels)
            pred_counts = np.bincount(preds, minlength=n_labels)
            x = np.arange(n_labels)
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
            plt.bar(x + width / 2, pred_counts, width, label="Predicted")
            plt.xlabel("Label")
            plt.ylabel("Count")
            plt.title(
                "SPR_BENCH Label Distribution\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_label_distribution.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating label distribution plot: {e}")
            plt.close()

    # Print evaluation metric
    print(f"Final Test Macro-F1: {exp.get('test_macroF1', 'N/A')}")
