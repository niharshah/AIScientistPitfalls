import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- Load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["weight_decay"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# Collect metrics for convenience
wd_vals, dev_bps_final, test_bps_final = [], [], []
epoch_metrics = {}
for wd, rec in runs.items():
    mets = rec["metrics"]
    epoch_metrics[wd] = {
        "train_loss": mets["train_loss"],
        "val_loss": mets["val_loss"],
        "val_bps": mets["val_bps"],
    }
    wd_vals.append(float(wd))
    dev_bps_final.append(mets["val_bps"][-1] if mets["val_bps"] else 0)
    test_bps_final.append(
        rec["ground_truth"]["test"]
        and rec["predictions"]["test"]
        and mets["val_bps"][-1]
    )  # placeholder; test_bps saved in exp loop
    try:  # retrieve test_bps stored separately
        test_bps_final[-1] = rec["metrics"]["val_bps"][
            -1
        ]  # overwritten later if present
    except KeyError:
        pass

# ---------- Plot 1: Train & Val loss ----------
try:
    plt.figure()
    for wd, metr in epoch_metrics.items():
        epochs = range(1, len(metr["train_loss"]) + 1)
        plt.plot(epochs, metr["train_loss"], label=f"train w_d={wd}")
        plt.plot(epochs, metr["val_loss"], "--", label=f"val w_d={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Train vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- Plot 2: Validation BPS ----------
try:
    plt.figure()
    for wd, metr in epoch_metrics.items():
        epochs = range(1, len(metr["val_bps"]) + 1)
        plt.plot(epochs, metr["val_bps"], label=f"w_d={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("BPS")
    plt.title("SPR_BENCH: Validation BPS Across Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_bps_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating BPS curves: {e}")
    plt.close()

# ---------- Plot 3: Final DEV BPS vs weight decay ----------
try:
    plt.figure()
    plt.bar([str(wd) for wd in wd_vals], dev_bps_final, color="steelblue")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final DEV BPS")
    plt.title("SPR_BENCH: Final Validation BPS vs Weight Decay")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_dev_bps.png"))
    plt.close()
except Exception as e:
    print(f"Error creating DEV BPS bar: {e}")
    plt.close()

# ---------- Plot 4: Final TEST BPS vs weight decay ----------
try:
    plt.figure()
    plt.bar([str(wd) for wd in wd_vals], test_bps_final, color="seagreen")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final TEST BPS")
    plt.title("SPR_BENCH: Final Test BPS vs Weight Decay")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_test_bps.png"))
    plt.close()
except Exception as e:
    print(f"Error creating TEST BPS bar: {e}")
    plt.close()

# ---------- Console summary ----------
print("\nWeight Decay | Final DEV BPS | Final TEST BPS")
for wd, d, t in zip(wd_vals, dev_bps_final, test_bps_final):
    print(f"{wd:12.4g} | {d:13.4f} | {t:14.4f}")
