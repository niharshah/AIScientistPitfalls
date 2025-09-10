import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------- Load data -----------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["NO_SYMBOLIC_BRANCH"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = None

if run:
    losses_tr = run["losses"]["train"]
    losses_val = run["losses"]["val"]
    swa_val = run["metrics"]["val"]
    swa_test = run["metrics"].get("test", None)
    preds = np.array(run["predictions"])
    gts = np.array(run["ground_truth"])
    epochs = np.arange(1, len(losses_tr) + 1)

    # ----------------------- Loss curves -----------------------
    try:
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train Loss")
        plt.plot(epochs, losses_val, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training vs. Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ----------------------- Validation metric curve -----------------------
    try:
        plt.figure()
        plt.plot(epochs, swa_val, marker="o", label="Validation SWA")
        if swa_test is not None:
            plt.axhline(
                swa_test, color="r", linestyle="--", label=f"Test SWA={swa_test:.3f}"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH – Validation SWA Across Epochs")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_validation_SWA.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating validation metric plot: {e}")
        plt.close()

    # ----------------------- Confusion matrix -----------------------
    try:
        classes = np.unique(np.concatenate([gts, preds]))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH – Confusion Matrix (Test Set)")
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xticks(classes)
        plt.yticks(classes)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ----------------------- Prediction vs. GT distribution -----------------------
    try:
        plt.figure()
        unique, counts_gt = np.unique(gts, return_counts=True)
        _, counts_pr = np.unique(preds, return_counts=True)
        bar_w = 0.35
        idx = np.arange(len(unique))
        plt.bar(idx - bar_w / 2, counts_gt, bar_w, label="Ground Truth")
        plt.bar(idx + bar_w / 2, counts_pr, bar_w, label="Predictions")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("SPR_BENCH – Class Distribution: GT vs. Predictions")
        plt.xticks(idx, unique)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_class_distribution.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating class distribution plot: {e}")
        plt.close()

    # ----------------------- Print evaluation metrics -----------------------
    print(
        f"Test Shape-Weighted Accuracy: {swa_test:.4f}"
        if swa_test is not None
        else "Test SWA not found."
    )
    print("Confusion Matrix:\n", cm)
