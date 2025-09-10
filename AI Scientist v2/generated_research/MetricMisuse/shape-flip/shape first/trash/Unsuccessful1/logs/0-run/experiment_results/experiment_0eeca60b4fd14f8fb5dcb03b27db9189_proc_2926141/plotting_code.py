import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- LOAD DATA --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
    epochs = spr["epochs"]
    tr_loss = spr["losses"]["train"]
    val_loss = spr["losses"]["val"]
    val_swa = spr["metrics"]["val"]["SWA"]
    val_cwa = spr["metrics"]["val"]["CWA"]
    val_hrg = spr["metrics"]["val"]["HRG"]
    test_swa = spr["metrics"]["test"]["SWA"]
    test_cwa = spr["metrics"]["test"]["CWA"]
    test_hrg = spr["metrics"]["test"]["HRG"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    epochs, tr_loss, val_loss, val_swa, val_cwa, val_hrg = [], [], [], [], [], []
    test_swa = test_cwa = test_hrg = None

# -------------------- PLOTS --------------------
# 1) Loss curve
try:
    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
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

# 2) SWA & CWA curve
try:
    plt.figure()
    plt.plot(epochs, val_swa, label="SWA")
    plt.plot(epochs, val_cwa, label="CWA")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR_BENCH: Validation SWA & CWA")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_SWA_CWA_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA/CWA curve: {e}")
    plt.close()

# 3) HRG curve
try:
    plt.figure()
    plt.plot(epochs, val_hrg, label="Validation HRG")
    if test_hrg is not None:
        plt.axhline(
            test_hrg, color="red", linestyle="--", label=f"Test HRG={test_hrg:.3f}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("HRG")
    plt.title("SPR_BENCH: Validation HRG (Test HRG dashed)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_HRG_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HRG curve: {e}")
    plt.close()

# -------------------- PRINT TEST METRICS --------------------
print(f"TEST METRICS -> SWA: {test_swa}, CWA: {test_cwa}, HRG: {test_hrg}")
