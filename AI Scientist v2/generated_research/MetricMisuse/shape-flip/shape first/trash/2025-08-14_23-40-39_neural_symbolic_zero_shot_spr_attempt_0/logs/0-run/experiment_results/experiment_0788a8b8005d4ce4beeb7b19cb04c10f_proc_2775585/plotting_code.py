import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
exp_path_candidates = [
    os.path.join(working_dir, "experiment_data.npy"),
    "experiment_data.npy",
]
experiment_data = None
for p in exp_path_candidates:
    if os.path.isfile(p):
        experiment_data = np.load(p, allow_pickle=True).item()
        break
if experiment_data is None:
    raise FileNotFoundError("experiment_data.npy not found in expected locations.")

spr_runs = experiment_data["epochs_tuning"]["SPR_BENCH"]
epoch_settings = sorted(spr_runs, key=lambda x: int(x))

# Gather metrics
metrics = {k: spr_runs[k]["metrics"] for k in epoch_settings}
final_dev = {k: spr_runs[k]["final_dev"]["bps"] for k in epoch_settings}
final_test = {k: spr_runs[k]["final_test"]["bps"] for k in epoch_settings}

# ---------------- Plot 1: Loss curves ----------------
try:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for k in epoch_settings:
        e = np.arange(1, len(metrics[k]["train_loss"]) + 1)
        axs[0].plot(e, metrics[k]["train_loss"], label=f"{k} ep")
        axs[1].plot(e, metrics[k]["val_loss"], label=f"{k} ep")
    axs[0].set_title("Training Loss")
    axs[1].set_title("Validation Loss")
    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle("SPR_BENCH Loss Curves | Left: Training Loss, Right: Validation Loss")
    plt.tight_layout()
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------------- Plot 2: Validation BPS per epoch ----------------
try:
    plt.figure(figsize=(6, 4))
    for k in epoch_settings:
        e = np.arange(1, len(metrics[k]["val_bps"]) + 1)
        plt.plot(e, metrics[k]["val_bps"], label=f"{k} ep")
    plt.xlabel("Epoch")
    plt.ylabel("BPS")
    plt.title("SPR_BENCH Validation BPS Across Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_bps_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val BPS plot: {e}")
    plt.close()

# ---------------- Plot 3: Final DEV BPS ----------------
try:
    plt.figure(figsize=(6, 4))
    plt.bar(
        range(len(final_dev)),
        list(final_dev.values()),
        tick_label=list(final_dev.keys()),
    )
    plt.xlabel("Training Epochs")
    plt.ylabel("BPS")
    plt.title("SPR_BENCH Final DEV BPS vs Epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_dev_bps.png"))
    plt.close()
except Exception as e:
    print(f"Error creating DEV BPS bar: {e}")
    plt.close()

# ---------------- Plot 4: Final TEST BPS ----------------
try:
    plt.figure(figsize=(6, 4))
    plt.bar(
        range(len(final_test)),
        list(final_test.values()),
        tick_label=list(final_test.keys()),
    )
    plt.xlabel("Training Epochs")
    plt.ylabel("BPS")
    plt.title("SPR_BENCH Final TEST BPS vs Epochs")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_test_bps.png"))
    plt.close()
except Exception as e:
    print(f"Error creating TEST BPS bar: {e}")
    plt.close()

# -------------- Print summary metrics --------------
print("Final BPS scores")
for k in epoch_settings:
    print(f"{k} epochs -> DEV BPS: {final_dev[k]:.4f} | TEST BPS: {final_test[k]:.4f}")
