import matplotlib.pyplot as plt
import numpy as np
import os

# --- paths ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load exp data ---
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["shape_blind"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = {}


# Helper for confusion matrix
def make_cm(true, pred, num_cls):
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm


# 1) Accuracy curves ---------------------------------------------------------
try:
    epochs = np.arange(1, len(ed["metrics"]["train"]) + 1)
    plt.figure()
    plt.plot(epochs, ed["metrics"]["train"], label="Train")
    plt.plot(epochs, ed["metrics"]["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("spr_bench Accuracy over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) Loss curves -------------------------------------------------------------
try:
    epochs = np.arange(1, len(ed["losses"]["train"]) + 1)
    plt.figure()
    plt.plot(epochs, ed["losses"]["train"], label="Train")
    plt.plot(epochs, ed["losses"]["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("spr_bench Loss over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) SWA curves --------------------------------------------------------------
try:
    epochs = np.arange(1, len(ed["swa"]["train"]) + 1)
    plt.figure()
    plt.plot(epochs, ed["swa"]["train"], label="Train")
    plt.plot(epochs, ed["swa"]["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("spr_bench SWA over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_swa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating swa plot: {e}")
    plt.close()

# 4) Confusion matrix (validation) ------------------------------------------
try:
    gt = ed["ground_truth"].get("val", [])
    pr = ed["predictions"].get("val", [])
    if gt and pr:
        num_cls = max(max(gt), max(pr)) + 1
        cm = make_cm(gt, pr, num_cls)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("spr_bench Confusion Matrix (Validation)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_cm_val.png"))
        plt.close()
except Exception as e:
    print(f"Error creating val confusion matrix: {e}")
    plt.close()

# 5) Confusion matrix (test) -------------------------------------------------
try:
    gt = ed["ground_truth"].get("test", [])
    pr = ed["predictions"].get("test", [])
    if gt and pr:
        num_cls = max(max(gt), max(pr)) + 1
        cm = make_cm(gt, pr, num_cls)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("spr_bench Confusion Matrix (Test)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "spr_bench_cm_test.png"))
        plt.close()
except Exception as e:
    print(f"Error creating test confusion matrix: {e}")
    plt.close()

# --- print final metrics ----------------------------------------------------
print("Final Test Metrics:", ed.get("test_metrics", "N/A"))
