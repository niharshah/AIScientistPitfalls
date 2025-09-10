import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["BinarySymFeat"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    epochs = range(1, len(exp["losses"]["train"]) + 1)

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, exp["losses"]["train"], label="Train Loss")
        plt.plot(epochs, exp["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot: {e}")
    finally:
        plt.close()

    # 2) Validation accuracy curve
    try:
        val_acc = exp["metrics"]["val"]
        if isinstance(val_acc, list) and any(v is not None for v in val_acc):
            vals = [v if v is not None else np.nan for v in val_acc]
            plt.figure()
            plt.plot(epochs, vals, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title("SPR_BENCH Validation SWA")
            fname = os.path.join(working_dir, "SPR_BENCH_val_accuracy.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        else:
            print("Validation accuracy data unavailable, skipping accuracy plot.")
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
    finally:
        plt.close()

    # 3) Ground-truth vs Prediction counts
    try:
        preds, gts = exp["predictions"], exp["ground_truth"]
        if preds and gts:
            classes = sorted(set(gts))
            gt_cnt = [gts.count(c) for c in classes]
            pr_cnt = [preds.count(c) for c in classes]

            x = np.arange(len(classes))
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, gt_cnt, width, label="Ground Truth")
            plt.bar(x + width / 2, pr_cnt, width, label="Predictions")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(
                "SPR_BENCH Class Distribution\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.xticks(x, classes)
            plt.legend()
            fname = os.path.join(working_dir, "SPR_BENCH_gt_vs_pred.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        else:
            print("Prediction data unavailable, skipping GT vs Pred plot.")
    except Exception as e:
        print(f"Error creating GT vs Pred plot: {e}")
    finally:
        plt.close()
