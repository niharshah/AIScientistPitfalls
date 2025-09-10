import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to safely fetch nested dict items
def _safe_get(dct, keys, default=None):
    for k in keys:
        if isinstance(dct, dict) and k in dct:
            dct = dct[k]
        else:
            return default
    return dct


# ------------------------------------------------------------------
# Plot 1: Accuracy vs alpha
try:
    alphas = _safe_get(
        experiment_data, ["ccp_alpha_tuning", "SPR_BENCH", "tested_alphas"]
    )
    train_acc = _safe_get(
        experiment_data, ["ccp_alpha_tuning", "SPR_BENCH", "metrics", "train"]
    )
    val_acc = _safe_get(
        experiment_data, ["ccp_alpha_tuning", "SPR_BENCH", "metrics", "val"]
    )

    if alphas and train_acc and val_acc:
        plt.figure()
        plt.plot(alphas, train_acc, marker="o", label="Train")
        plt.plot(alphas, val_acc, marker="s", label="Validation")
        plt.xlabel("ccp_alpha")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Train vs Validation Accuracy vs ccp_alpha")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_vs_alpha.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        raise ValueError("Required accuracy data missing.")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 2: Loss vs alpha
try:
    alphas = _safe_get(
        experiment_data, ["ccp_alpha_tuning", "SPR_BENCH", "tested_alphas"]
    )
    train_loss = _safe_get(
        experiment_data, ["ccp_alpha_tuning", "SPR_BENCH", "losses", "train"]
    )
    val_loss = _safe_get(
        experiment_data, ["ccp_alpha_tuning", "SPR_BENCH", "losses", "val"]
    )

    if alphas and train_loss and val_loss:
        plt.figure()
        plt.plot(alphas, train_loss, marker="o", label="Train")
        plt.plot(alphas, val_loss, marker="s", label="Validation")
        plt.xlabel("ccp_alpha")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss vs ccp_alpha")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_vs_alpha.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        raise ValueError("Required loss data missing.")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Plot 3: Confusion matrix on test data
try:
    y_true = np.array(
        _safe_get(experiment_data, ["ccp_alpha_tuning", "SPR_BENCH", "ground_truth"])
    )
    y_pred = np.array(
        _safe_get(experiment_data, ["ccp_alpha_tuning", "SPR_BENCH", "predictions"])
    )

    if y_true.size and y_pred.size:
        # Compute confusion matrix (binary)
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1

        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.title("SPR_BENCH: Confusion Matrix on Test Set")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    else:
        raise ValueError("Ground truth or predictions missing.")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print evaluation metrics if available
try:
    test_acc = (y_true == y_pred).mean() if y_true.size else None

    # Recompute SEFA if losses not stored
    def decision_tree_single_pred(clf, x_row):  # placeholder
        return 0

    sefa = _safe_get(
        experiment_data, ["ccp_alpha_tuning", "SPR_BENCH", "sefa"]
    )  # might not exist
    print(
        f"Test Accuracy: {test_acc:.4f}"
        if test_acc is not None
        else "Test Accuracy: N/A"
    )
    print(f"SEFA: {sefa:.4f}" if sefa is not None else "SEFA: N/A")
except Exception as e:
    print(f"Error computing evaluation metrics: {e}")
