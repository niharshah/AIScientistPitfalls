import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr:
    losses_tr = spr["losses"]["train"]
    losses_val = spr["losses"]["val"]
    swa_tr = spr["metrics"]["train"]
    swa_val = spr["metrics"]["val"]
    preds = np.array(spr["predictions"])
    gts = np.array(spr["ground_truth"])

    # 1. Loss curves --------------------------------------------------------
    try:
        epochs = np.arange(1, len(losses_tr) + 1)
        plt.figure(figsize=(5, 3))
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, linestyle="--", label="Validation")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves: {e}")
        plt.close()

    # 2. SWA curves ---------------------------------------------------------
    try:
        epochs = np.arange(1, len(swa_tr) + 1)
        plt.figure(figsize=(5, 3))
        plt.plot(epochs, swa_tr, label="Train")
        plt.plot(epochs, swa_val, linestyle="--", label="Validation")
        plt.title("SPR_BENCH Shape-Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_swa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting SWA curves: {e}")
        plt.close()

    # 3. Confusion matrix ---------------------------------------------------
    try:
        from collections import Counter

        n_cls = int(max(gts.max(), preds.max())) + 1
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[gt, pr] += 1
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, cmap="Blues")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predictions")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        plt.close()

    # print evaluation summary ---------------------------------------------
    val_swa_final = swa_val[-1] if swa_val else None
    test_swa = (
        (cm.diagonal().sum() / cm.sum()) if "cm" in locals() and cm.sum() else None
    )
    print(f"Final Validation SWA: {val_swa_final:.4f}" if val_swa_final else "")
    print(f"Test SWA (from confusion matrix): {test_swa:.4f}" if test_swa else "")
