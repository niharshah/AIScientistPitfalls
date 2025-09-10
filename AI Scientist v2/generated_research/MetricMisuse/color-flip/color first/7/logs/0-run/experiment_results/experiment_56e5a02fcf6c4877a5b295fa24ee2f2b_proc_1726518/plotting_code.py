import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data ---------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["epochs_tuning"]["SPR_BENCH"]
    epochs = exp["epochs"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)


# helper to fetch series safely
def get_series(key, split):
    return (
        [m[key] if m is not None else None for m in exp["metrics"][split]]
        if key in exp["metrics"][split][0]
        else []
    )


# --------- 1. Loss curve ---------------
try:
    plt.figure()
    tr_loss = exp["losses"]["train"]
    val_loss = exp["losses"]["val"]
    plt.plot(epochs, tr_loss, label="Train")
    if any(v is not None for v in val_loss):
        plt.plot(
            epochs, [v if v is not None else np.nan for v in val_loss], label="Val"
        )
    plt.title("SPR_BENCH: Loss Curve (Train vs Validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Loss curve: {e}")
    plt.close()

# --------- 2. Complexity-Weighted Acc ----------------
try:
    plt.figure()
    tr = [m["cpx"] for m in exp["metrics"]["train"]]
    val = [m["cpx"] for m in exp["metrics"]["val"]]
    plt.plot(epochs, tr, label="Train")
    plt.plot(epochs, val, label="Val")
    plt.title("SPR_BENCH: Complexity-Weighted Accuracy (CpxWA)")
    plt.xlabel("Epoch")
    plt.ylabel("CpxWA")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_cpxwa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CpxWA curve: {e}")
    plt.close()

# --------- 3. Color-Weighted Acc ---------------------
try:
    plt.figure()
    tr = [m["cwa"] for m in exp["metrics"]["train"]]
    val = [m["cwa"] for m in exp["metrics"]["val"]]
    plt.plot(epochs, tr, label="Train")
    plt.plot(epochs, val, label="Val")
    plt.title("SPR_BENCH: Color-Weighted Accuracy (CWA)")
    plt.xlabel("Epoch")
    plt.ylabel("CWA")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_cwa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA curve: {e}")
    plt.close()

# --------- 4. Shape-Weighted Acc ---------------------
try:
    plt.figure()
    tr = [m["swa"] for m in exp["metrics"]["train"]]
    val = [m["swa"] for m in exp["metrics"]["val"]]
    plt.plot(epochs, tr, label="Train")
    plt.plot(epochs, val, label="Val")
    plt.title("SPR_BENCH: Shape-Weighted Accuracy (SWA)")
    plt.xlabel("Epoch")
    plt.ylabel("SWA")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_swa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

print(f"Finished plotting. Figures saved to {working_dir}")
