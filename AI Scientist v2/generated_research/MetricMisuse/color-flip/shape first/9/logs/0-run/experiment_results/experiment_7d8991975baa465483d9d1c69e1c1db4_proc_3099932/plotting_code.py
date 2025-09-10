import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    lr_dict = experiment_data.get("learning_rate", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    lr_dict = {}

# ---------- per-learning-rate plots ----------
for lr_key, lr_data in lr_dict.items():
    # 1) Loss curves ----------------------------------------------------------
    try:
        train_losses = lr_data["losses"].get("train", [])
        val_losses = lr_data["losses"].get("val", [])
        if train_losses or val_losses:  # plot only if at least one list is non-empty
            plt.figure()
            if train_losses:
                plt.plot(train_losses, label="train")
            if val_losses:
                plt.plot(val_losses, label="val")
            plt.title(f"SPR_BENCH Loss Curve (lr={lr_key})")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = f"SPR_BENCH_loss_lr_{lr_key}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for lr={lr_key}: {e}")
        plt.close()

    # 2) Validation metric curves --------------------------------------------
    try:
        val_metrics = lr_data["metrics"].get("val", [])
        if val_metrics:
            plt.figure()
            plt.plot(val_metrics, marker="o")
            plt.title(f"SPR_BENCH CWA-2D Curve (lr={lr_key})")
            plt.xlabel("Epoch")
            plt.ylabel("CWA-2D")
            plt.grid(True)
            fname = f"SPR_BENCH_cwa_lr_{lr_key}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating metric plot for lr={lr_key}: {e}")
        plt.close()

# ---------- aggregate comparison of final metrics ---------------------------
try:
    lrs, finals = [], []
    for lr_key, lr_data in lr_dict.items():
        vals = lr_data["metrics"].get("val", [])
        if vals:  # take last epoch value
            lrs.append(lr_key)
            finals.append(vals[-1])
    if finals:
        plt.figure()
        plt.bar(lrs, finals)
        plt.title("SPR_BENCH Final CWA-2D vs Learning Rate")
        plt.xlabel("Learning Rate")
        plt.ylabel("Final CWA-2D")
        fname = "SPR_BENCH_final_cwa_comparison.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating aggregate plot: {e}")
    plt.close()

print(f"Finished plotting. Figures saved to {working_dir}")
