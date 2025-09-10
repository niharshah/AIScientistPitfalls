import matplotlib.pyplot as plt
import numpy as np
import os

# --- paths ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load experiment dict ---
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds = experiment_data.get("token_only", {}).get("spr_bench", {})


# Helper to save plot safely
def safe_save(fig, fname):
    fpath = os.path.join(working_dir, fname)
    fig.savefig(fpath)
    plt.close(fig)


# 1) Accuracy curve
try:
    tr = ds["metrics"]["train"]
    val = ds["metrics"]["val"]
    epochs = range(1, len(tr) + 1)
    fig = plt.figure()
    plt.plot(epochs, tr, label="Train")
    plt.plot(epochs, val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR-BENCH Accuracy Curve")
    plt.legend()
    safe_save(fig, "spr_bench_accuracy_curve.png")
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) Loss curve
try:
    tr = ds["losses"]["train"]
    val = ds["losses"]["val"]
    epochs = range(1, len(tr) + 1)
    fig = plt.figure()
    plt.plot(epochs, tr, label="Train")
    plt.plot(epochs, val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR-BENCH Loss Curve")
    plt.legend()
    safe_save(fig, "spr_bench_loss_curve.png")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) Shape-Weighted Accuracy curve
try:
    tr = ds["swa"]["train"]
    val = ds["swa"]["val"]
    epochs = range(1, len(tr) + 1)
    fig = plt.figure()
    plt.plot(epochs, tr, label="Train")
    plt.plot(epochs, val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR-BENCH SWA Curve")
    plt.legend()
    safe_save(fig, "spr_bench_swa_curve.png")
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# 4) Confusion matrix on validation set
try:
    preds = np.array(ds["predictions"]["val"])
    gts = np.array(ds["ground_truth"]["val"])
    n_cls = max(gts.max(), preds.max()) + 1
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1
    fig = plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR-BENCH Confusion Matrix (Validation)")
    safe_save(fig, "spr_bench_confusion_matrix_val.png")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# --- print stored test metrics ---
try:
    tst = ds["test_metrics"]
    print(
        f"Test metrics ->  Loss: {tst['loss']:.4f} | "
        f"Accuracy: {tst['acc']:.3f} | SWA: {tst['swa']:.3f}"
    )
except Exception as e:
    print(f"Could not print test metrics: {e}")
