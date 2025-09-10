import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp_key = "NoContrastivePretraining"
dset_key = "SPR_BENCH"
if exp_key in experiment_data and dset_key in experiment_data[exp_key]:
    data = experiment_data[exp_key][dset_key]
    losses_tr = data["losses"]["train"]
    losses_val = data["losses"]["val"]
    cwca_tr = data["metrics"]["train"]
    cwca_val = data["metrics"]["val"]
    cwca_test = data["metrics"]["test"][0] if data["metrics"]["test"] else None
    y_pred = np.array(data["predictions"])
    y_true = np.array(data["ground_truth"])
else:
    print("Required keys not found in experiment_data.")
    losses_tr = losses_val = cwca_tr = cwca_val = []
    cwca_test = None
    y_pred = y_true = np.array([])

# ------------------------------------------------------------------
# 1) Loss curves
try:
    plt.figure()
    epochs = range(1, len(losses_tr) + 1)
    plt.plot(epochs, losses_tr, label="Train")
    plt.plot(epochs, losses_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss Curves – SPR_BENCH")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) CWCA curves (+ test point)
try:
    plt.figure()
    plt.plot(epochs, cwca_tr, label="Train")
    plt.plot(epochs, cwca_val, label="Validation")
    if cwca_test is not None:
        plt.scatter(
            [epochs[-1]], [cwca_test], color="red", label=f"Test ({cwca_test:.3f})"
        )
    plt.xlabel("Epoch")
    plt.ylabel("CWCA")
    plt.title("CWCA Curves – SPR_BENCH")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_cwca_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWCA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Confusion matrix heat-map
try:
    if y_true.size and y_pred.size:
        n_lbl = int(max(y_true.max(), y_pred.max()) + 1)
        cm = np.zeros((n_lbl, n_lbl), int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title(
            "Confusion Matrix – SPR_BENCH\nLeft: Ground Truth (rows), Right: Predictions (cols)"
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(n_lbl):
            for j in range(n_lbl):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.colorbar()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------------------------------------------------------
if cwca_test is not None:
    print(f"Final Test CWCA: {cwca_test:.4f}")
