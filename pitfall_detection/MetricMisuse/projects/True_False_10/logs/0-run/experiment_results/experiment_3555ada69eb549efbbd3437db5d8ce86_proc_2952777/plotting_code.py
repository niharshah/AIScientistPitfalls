import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)

# Navigate to the only run we have
try:
    d = experiment_data["no_symbolic_features"]["SPR_BENCH"]
    epochs = np.array(d["epochs"])
    train_loss = np.array(d["losses"]["train"])
    val_loss = np.array(d["losses"]["val"])
    val_swa = np.array(d["metrics"]["val"])
    y_true = np.array(d["ground_truth"][-1])
    y_pred = np.array(d["predictions"][-1])
except Exception as e:
    print(f"Error extracting data: {e}")
    exit(0)

# -------------------------------------------------------------------------
# 1) Loss curves
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 2) Validation SWA
try:
    plt.figure()
    plt.plot(epochs, val_swa, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH: Validation SWA over Epochs")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_SWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 3) Confusion matrix for final epoch
try:
    # Compute 2x2 confusion counts
    num_cls = len(np.unique(np.concatenate([y_true, y_pred])))
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SPR_BENCH: Confusion Matrix (Final Epoch)")
    for i in range(num_cls):
        for j in range(num_cls):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
