import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load experiment log -------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)

spr = experiment_data.get("SPR_BENCH", {})
if not spr:
    print("SPR_BENCH entry not found in experiment_data.")
    exit(0)

loss_tr = spr["losses"].get("train", [])
loss_val = spr["losses"].get("val", [])
swa_val = spr["metrics"].get("val", [])
swa_test = spr["metrics"].get("test", None)

# -------------------- PLOT 1: loss curves -------------------------
try:
    if loss_tr and loss_val:
        epochs = np.arange(1, len(loss_tr) + 1)
        plt.figure()
        plt.plot(epochs, loss_tr, "--o", label="Train")
        plt.plot(epochs, loss_val, "-s", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------------- PLOT 2: validation SWA ----------------------
try:
    if swa_val:
        epochs = np.arange(1, len(swa_val) + 1)
        plt.figure()
        plt.plot(epochs, swa_val, "-^", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH: Validation SWA Across Epochs")
        fname = os.path.join(working_dir, "SPR_BENCH_val_SWA_curve.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating validation SWA plot: {e}")
    plt.close()

# -------------------- PLOT 3: final test SWA ----------------------
try:
    if swa_test is not None:
        plt.figure()
        plt.bar(["Test"], [swa_test], color="steelblue")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH: Final Test SWA")
        fname = os.path.join(working_dir, "SPR_BENCH_test_SWA_bar.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating test SWA bar plot: {e}")
    plt.close()

# ---------------------- print metric ------------------------------
if swa_test is not None:
    print(f"Final Test SWA: {swa_test:.4f}")
