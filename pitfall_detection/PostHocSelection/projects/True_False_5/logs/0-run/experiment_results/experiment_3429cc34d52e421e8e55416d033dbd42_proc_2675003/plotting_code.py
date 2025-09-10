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
    data = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

# ------------------------------------------------------------------
# 1. Loss curves ----------------------------------------------------
try:
    tr_losses = [x[1] for x in data["losses"]["train"]]
    val_losses = [x[1] for x in data["losses"]["val"]]
    epochs = np.arange(1, len(tr_losses) + 1)

    plt.figure()
    plt.plot(epochs, tr_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2. Validation RCWA ------------------------------------------------
try:
    val_rcwa = [x[1] for x in data["metrics"]["val"]]
    epochs = np.arange(1, len(val_rcwa) + 1)

    plt.figure()
    plt.plot(epochs, val_rcwa, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("RCWA")
    plt.title("SPR_BENCH: Validation RCWA over Epochs")
    fname = os.path.join(working_dir, "SPR_BENCH_RCWA_curve.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating RCWA curve: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3. Confusion matrix ----------------------------------------------
try:
    y_true = np.array(data["ground_truth"])
    y_pred = np.array(data["predictions"])
    num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("SPR_BENCH: Confusion Matrix (True vs Predicted)")
    plt.xticks(np.arange(num_classes))
    plt.yticks(np.arange(num_classes))
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
