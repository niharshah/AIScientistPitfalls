import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------------- #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------- load data ---------------------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

subtree = experiment_data.get("NUM_EPOCHS", {}).get("SPR_BENCH", {})
if not subtree:
    print("No SPR_BENCH data found.")
    exit()

# ------------------------ identify best config --------------------------- #
best_cfg, best_val = None, -1e9
val_mcc_per_cfg, test_mcc_per_cfg = {}, {}
for cfg, rec in subtree.items():
    max_val = max(rec["metrics"]["val_MCC"])
    val_mcc_per_cfg[cfg] = max_val
    test_mcc_per_cfg[cfg] = rec["metrics"].get("test_MCC", np.nan)
    if max_val > best_val:
        best_val, best_cfg = max_val, cfg

print(
    f"Best configuration: {best_cfg} | best val_MCC: {best_val:.4f} | "
    f"test_MCC: {test_mcc_per_cfg[best_cfg]:.4f}"
)

best_run = subtree[best_cfg]
epochs = best_run["epochs"]

# ------------------------------- plots ----------------------------------- #
# 1) Loss curves for best config
try:
    plt.figure()
    plt.plot(epochs, best_run["losses"]["train"], label="Train Loss")
    plt.plot(epochs, best_run["losses"]["val"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"SPR_BENCH Training vs Validation Loss\n(Best Config: {best_cfg})")
    plt.savefig(os.path.join(working_dir, f"{best_cfg}_loss_curve_SPR_BENCH.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Validation MCC curve for best config
try:
    plt.figure()
    plt.plot(epochs, best_run["metrics"]["val_MCC"], label="Validation MCC")
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.legend()
    plt.title(f"SPR_BENCH Validation MCC over Epochs\n(Best Config: {best_cfg})")
    plt.savefig(os.path.join(working_dir, f"{best_cfg}_valMCC_curve_SPR_BENCH.png"))
    plt.close()
except Exception as e:
    print(f"Error creating MCC curve: {e}")
    plt.close()

# 3) Bar plot: best val_MCC per config
try:
    plt.figure()
    cfgs = list(val_mcc_per_cfg.keys())
    vals = [val_mcc_per_cfg[c] for c in cfgs]
    plt.bar(cfgs, vals, color="skyblue")
    plt.ylabel("Best Validation MCC")
    plt.title(
        "SPR_BENCH Best Validation MCC per Epoch Setting\nLeft: Config, Right: MCC"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "valMCC_comparison_SPR_BENCH.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val_MCC comparison: {e}")
    plt.close()

# 4) Bar plot: test MCC per config
try:
    plt.figure()
    cfgs = list(test_mcc_per_cfg.keys())
    tests = [test_mcc_per_cfg[c] for c in cfgs]
    plt.bar(cfgs, tests, color="lightgreen")
    plt.ylabel("Test MCC")
    plt.title("SPR_BENCH Test MCC per Epoch Setting\nLeft: Config, Right: MCC")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "testMCC_comparison_SPR_BENCH.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test_MCC comparison: {e}")
    plt.close()
