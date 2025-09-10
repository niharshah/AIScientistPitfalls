import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment log ---------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)

bench_key = "SPR_BENCH"
if bench_key not in experiment_data:
    print("SPR_BENCH data not found.")
    exit(0)

bench = experiment_data[bench_key]
loss_tr = bench["losses"].get("train", [])
loss_val = bench["losses"].get("val", [])
swa_val = bench["metrics"].get("val", [])

# --------------------- PLOT 1: loss curves ---------------------
try:
    if loss_tr and loss_val:
        epochs = list(range(1, len(loss_tr) + 1))
        plt.figure()
        plt.plot(epochs, loss_tr, "--o", label="Train")
        plt.plot(epochs, loss_val, "-s", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# --------------------- PLOT 2: validation SWA ------------------
try:
    if swa_val:
        epochs = list(range(1, len(swa_val) + 1))
        plt.figure()
        plt.plot(epochs, swa_val, marker="d", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.title("SPR_BENCH: Validation Shape-Weighted Accuracy")
        fname = os.path.join(working_dir, "SPR_BENCH_val_SWA.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating SWA curve plot: {e}")
    plt.close()

# --------------- PLOT 3: Confusion Matrix if present ----------
try:
    preds = bench["predictions"].get("FiLM", [])
    gts = bench["ground_truth"].get("FiLM", [])
    if preds and gts:
        preds = np.array(preds)
        gts = np.array(gts)
        n_cls = int(max(preds.max(), gts.max())) + 1
        if n_cls <= 15:  # keep the matrix readable
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("SPR_BENCH Confusion Matrix (Test Set)")
            plt.colorbar()
            plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
            plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# --------------------- print final metric ----------------------
test_swa = bench["meta"].get("SWA_test_FiLM", None)
if test_swa is not None:
    print(f"Final TEST SWA (FiLM): {test_swa:.4f}")
