import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["weight_decay"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

decays = np.array(ed["decay_values"])
train_loss = np.array(ed["losses"]["train"])
val_loss = np.array(ed["losses"]["val"])
test_loss = np.array(ed["losses"]["test"])
val_crwa = np.array([m["CRWA"] for m in ed["metrics"]["val"]])
test_crwa = np.array([m["CRWA"] for m in ed["metrics"]["test"]])
val_swa = np.array([m["SWA"] for m in ed["metrics"]["val"]])
test_swa = np.array([m["SWA"] for m in ed["metrics"]["test"]])
val_cwa = np.array([m["CWA"] for m in ed["metrics"]["val"]])
test_cwa = np.array([m["CWA"] for m in ed["metrics"]["test"]])


# ---------------- plotting helpers ----------------
def save_fig(name):
    plt.savefig(os.path.join(working_dir, name), dpi=150)
    plt.close()


# 1) Loss curves
try:
    plt.figure()
    plt.plot(decays, train_loss, "o-", label="Train")
    plt.plot(decays, val_loss, "s--", label="Validation")
    plt.plot(decays, test_loss, "d:", label="Test")
    plt.xlabel("Weight Decay")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss vs Weight Decay")
    plt.legend()
    plt.xscale("log")
    save_fig("SPR_BENCH_loss_vs_weight_decay.png")
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) CRWA
try:
    plt.figure()
    plt.plot(decays, val_crwa, "s--", label="Validation")
    plt.plot(decays, test_crwa, "d:", label="Test")
    plt.xlabel("Weight Decay")
    plt.ylabel("CRWA")
    plt.title("SPR_BENCH CRWA vs Weight Decay")
    plt.xscale("log")
    plt.legend()
    save_fig("SPR_BENCH_CRWA_vs_weight_decay.png")
except Exception as e:
    print(f"Error creating CRWA plot: {e}")
    plt.close()

# 3) SWA
try:
    plt.figure()
    plt.plot(decays, val_swa, "s--", label="Validation")
    plt.plot(decays, test_swa, "d:", label="Test")
    plt.xlabel("Weight Decay")
    plt.ylabel("SWA")
    plt.title("SPR_BENCH SWA vs Weight Decay")
    plt.xscale("log")
    plt.legend()
    save_fig("SPR_BENCH_SWA_vs_weight_decay.png")
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# 4) CWA
try:
    plt.figure()
    plt.plot(decays, val_cwa, "s--", label="Validation")
    plt.plot(decays, test_cwa, "d:", label="Test")
    plt.xlabel("Weight Decay")
    plt.ylabel("CWA")
    plt.title("SPR_BENCH CWA vs Weight Decay")
    plt.xscale("log")
    plt.legend()
    save_fig("SPR_BENCH_CWA_vs_weight_decay.png")
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# ---------------- best configuration summary ----------------
best_idx = int(np.argmax(val_crwa))
print(f"Best validation CRWA achieved at weight_decay={decays[best_idx]:.1e}")
print(
    f"   Val  CRWA/SWA/CWA: {val_crwa[best_idx]:.4f}/{val_swa[best_idx]:.4f}/{val_cwa[best_idx]:.4f}"
)
print(
    f"   Test CRWA/SWA/CWA: {test_crwa[best_idx]:.4f}/{test_swa[best_idx]:.4f}/{test_cwa[best_idx]:.4f}"
)

# Optional: full metric table
print(
    "\nFull metric table (decay, val_CRWA, test_CRWA, val_SWA, test_SWA, val_CWA, test_CWA):"
)
for d, vc, tc, vs, ts, vca, tca in zip(
    decays, val_crwa, test_crwa, val_swa, test_swa, val_cwa, test_cwa
):
    print(f"{d:.1e}\t{vc:.4f}\t{tc:.4f}\t{vs:.4f}\t{ts:.4f}\t{vca:.4f}\t{tca:.4f}")
