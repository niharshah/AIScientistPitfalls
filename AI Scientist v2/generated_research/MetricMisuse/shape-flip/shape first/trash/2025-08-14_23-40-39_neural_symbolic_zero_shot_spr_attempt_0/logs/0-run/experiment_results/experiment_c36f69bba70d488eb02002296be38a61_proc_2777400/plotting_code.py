import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment results ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

lr_block = experiment_data.get("learning_rate", {})
final_bps = {}

# ---------- per-LR training / validation loss curves ----------
for lr_key, lr_dict in lr_block.items():
    try:
        metrics = lr_dict["metrics"]
        epochs = list(range(1, len(metrics["train_loss"]) + 1))
        plt.figure()
        plt.plot(epochs, metrics["train_loss"], label="Train Loss")
        plt.plot(epochs, metrics["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH | LR={lr_key} | Training vs Validation Loss")
        plt.legend()
        fname = f"SPR_BENCH_lr_{lr_key}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        # record final BPS
        final_bps[lr_key] = metrics["val_bps"][-1] if metrics["val_bps"] else None
    except Exception as e:
        print(f"Error creating loss curve for LR {lr_key}: {e}")
        plt.close()

# ---------- bar plot of final BPS ----------
try:
    plt.figure()
    keys = list(final_bps.keys())
    vals = [final_bps[k] for k in keys]
    plt.bar(keys, vals, color="skyblue")
    plt.ylabel("Final Validation BPS")
    plt.title("SPR_BENCH | Final BPS vs Learning Rate")
    fname = "SPR_BENCH_final_bps_comparison.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating BPS bar plot: {e}")
    plt.close()

# ---------- print numeric results ----------
print("Final validation BPS per learning rate:")
for k, v in final_bps.items():
    print(f"  LR={k}: {v:.4f}" if v is not None else f"  LR={k}: N/A")
