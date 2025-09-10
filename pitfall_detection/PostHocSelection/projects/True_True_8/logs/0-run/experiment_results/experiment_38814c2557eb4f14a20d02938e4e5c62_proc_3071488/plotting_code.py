import matplotlib.pyplot as plt
import numpy as np
import os

# ---- working dir ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load data ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

lr_sweep = experiment_data.get("lr_sweep", {})
lrs = sorted(lr_sweep.keys())  # keep a consistent order


# helper to extract metric list -> two arrays
def metric_xy(lr_key, metric_name):
    vals = lr_sweep[lr_key]["metrics"][metric_name]
    if not vals:
        return [], []
    x, y = zip(*vals)
    return list(x), list(y)


final_acs = {}

# ---- Figure 1: train/val loss curves ----
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for lr_key in lrs[:]:  # all lrs but still <=4, so OK
        x_tr, y_tr = metric_xy(lr_key, "train_loss")
        x_v, y_v = metric_xy(lr_key, "val_loss")
        axes[0].plot(x_tr, y_tr, label=lr_key)
        axes[1].plot(x_v, y_v, label=lr_key)
    axes[0].set_title("Train Loss")
    axes[1].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Loss")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Synthetic SPR Dataset – Left: Train, Right: Validation Loss")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(working_dir, "spr_loss_curves.png")
    plt.savefig(out_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---- Figure 2: validation ACS curves ----
try:
    plt.figure(figsize=(6, 4))
    for lr_key in lrs:
        x_ac, y_ac = metric_xy(lr_key, "val_ACS")
        plt.plot(x_ac, y_ac, marker="o", label=lr_key)
        if x_ac:
            final_acs[lr_key] = y_ac[-1]
    plt.title("Synthetic SPR Dataset – Validation ACS over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("ACS")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(working_dir, "spr_val_ACS_curves.png")
    plt.savefig(out_path)
    plt.close()
except Exception as e:
    print(f"Error creating ACS curves: {e}")
    plt.close()

# ---- Figure 3: final ACS bar chart ----
try:
    if final_acs:
        plt.figure(figsize=(6, 4))
        lr_names = list(final_acs.keys())
        acs_vals = [final_acs[k] for k in lr_names]
        plt.bar(range(len(lr_names)), acs_vals)
        plt.xticks(range(len(lr_names)), lr_names, rotation=45)
        plt.ylabel("Final Validation ACS")
        plt.title("Synthetic SPR Dataset – Final ACS per Learning-Rate")
        plt.tight_layout()
        out_path = os.path.join(working_dir, "spr_final_ACS_bar.png")
        plt.savefig(out_path)
    plt.close()
except Exception as e:
    print(f"Error creating final ACS bar: {e}")
    plt.close()

# ---- print evaluation metrics ----
if final_acs:
    print("Final Validation ACS per learning rate:")
    for k, v in final_acs.items():
        print(f"  {k}: {v:.4f}")
