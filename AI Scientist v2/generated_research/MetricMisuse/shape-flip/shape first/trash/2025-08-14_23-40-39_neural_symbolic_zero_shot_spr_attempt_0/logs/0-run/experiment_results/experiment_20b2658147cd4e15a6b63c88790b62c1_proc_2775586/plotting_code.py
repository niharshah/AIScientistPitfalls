import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# helper for LR keys in the right order
lr_dict = experiment_data.get("learning_rate", {})
lr_keys = sorted(
    lr_dict.keys(), key=lambda x: float(x.replace("e", "e+"))
)  # "1e-3" etc.

final_bps = {}

# plot curves for each LR (max 5 -> already satisfied)
for lr_key in lr_keys:
    try:
        md = lr_dict[lr_key]["metrics"]
        train_loss = md["train_loss"]
        val_loss = md["val_loss"]
        epochs = range(1, len(train_loss) + 1)

        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SPR_BENCH Training vs Validation Loss\nLR={lr_key}")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_lr_{lr_key}_loss_curve.png")
        plt.savefig(fname)
        plt.close()

        # store last BPS
        final_bps[lr_key] = md["val_bps"][-1] if md["val_bps"] else None
    except Exception as e:
        print(f"Error creating loss curve for LR {lr_key}: {e}")
        plt.close()

# bar chart comparing final BPS across LRs
try:
    keys, bps_vals = zip(*[(k, v) for k, v in final_bps.items() if v is not None])
    plt.figure()
    plt.bar(keys, bps_vals, color="skyblue")
    plt.ylabel("Final Validation BPS")
    plt.title("SPR_BENCH Final BPS per Learning Rate")
    fname = os.path.join(working_dir, "SPR_BENCH_lr_sweep_BPS.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating BPS comparison plot: {e}")
    plt.close()

# print best LR summary
if final_bps:
    best_lr = max(final_bps, key=lambda k: final_bps[k])
    print(
        f"Best LR based on final validation BPS: {best_lr} (BPS={final_bps[best_lr]:.4f})"
    )
