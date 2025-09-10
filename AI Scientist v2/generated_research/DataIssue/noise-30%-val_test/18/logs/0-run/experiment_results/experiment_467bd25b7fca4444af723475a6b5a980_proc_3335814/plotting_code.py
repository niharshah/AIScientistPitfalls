import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup ---------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data ------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr, experiment_data = {}, {}

# Early exit if nothing loaded
if not spr:
    exit()

epochs = list(range(1, len(spr["losses"]["train"]) + 1))
loss_tr, loss_val = spr["losses"]["train"], spr["losses"]["val"]
f1_tr = [m["f1"] for m in spr["metrics"]["train"]]
f1_val = [m["f1"] for m in spr["metrics"]["val"]]
mcc_tr = [m["mcc"] for m in spr["metrics"]["train"]]
mcc_val = [m["mcc"] for m in spr["metrics"]["val"]]
preds, gts = np.array(spr["predictions"]), np.array(spr["ground_truth"])

# -------- 1. Loss curves ------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Validation", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# -------- 2. F1 curves --------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, f1_tr, label="Train")
    plt.plot(epochs, f1_val, label="Validation", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training vs Validation F1")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# -------- 3. MCC curves -------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, mcc_tr, label="Train")
    plt.plot(epochs, mcc_val, label="Validation", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.title("SPR_BENCH: Training vs Validation MCC")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_mcc_curves.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating MCC curves: {e}")
    plt.close()

# -------- 4. Confusion matrix -------------------------------------------------
try:
    cm = np.zeros((2, 2), dtype=int)
    for g, p in zip(gts, preds):
        cm[g, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
    plt.colorbar()
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------- 5. Print test metrics ----------------------------------------------
try:
    print("Test metrics:", spr["test_metrics"])
except Exception as e:
    print(f"Error printing test metrics: {e}")
