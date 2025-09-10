import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "SPR_BENCH"
lr_dict = experiment_data.get("learning_rate", {})


# helper: get a nice lr label
def lr_label(lr_key):
    return lr_key.replace("p", ".")


# -------------- figure 1 : pretrain loss curves ----------
try:
    plt.figure()
    for lr_key, d in lr_dict.items():
        losses = d[ds_name]["losses"]["pretrain"]
        plt.plot(range(1, len(losses) + 1), losses, label=f"lr={lr_label(lr_key)}")
    plt.title(f"{ds_name} Pre-training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_pretrain_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating pretrain loss figure: {e}")
    plt.close()

# -------------- figure 2 : train / val loss curves -------
try:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for lr_key, d in lr_dict.items():
        train = d[ds_name]["losses"]["train"]
        val = d[ds_name]["losses"]["val"]
        ax[0].plot(range(1, len(train) + 1), train, label=f"lr={lr_label(lr_key)}")
        ax[1].plot(range(1, len(val) + 1), val, label=f"lr={lr_label(lr_key)}")
    ax[0].set_title("Train")
    ax[1].set_title("Validation")
    for a in ax:
        a.set_xlabel("Epoch")
        a.set_ylabel("Loss")
    ax[1].legend()
    fig.suptitle(f"{ds_name} Fine-tune Loss\nLeft: Train, Right: Validation")
    fname = os.path.join(working_dir, f"{ds_name}_finetune_loss_curves.png")
    plt.savefig(fname)
    plt.close(fig)
except Exception as e:
    print(f"Error creating train/val loss figure: {e}")
    plt.close()

# -------------- figure 3 : metrics curves -----------------
try:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    metrics_names = ["SWA", "CWA", "SCHM"]
    for idx, m in enumerate(metrics_names):
        for lr_key, d in lr_dict.items():
            vals = d[ds_name]["metrics"][m]
            ax[idx].plot(range(1, len(vals) + 1), vals, label=f"lr={lr_label(lr_key)}")
        ax[idx].set_title(m)
        ax[idx].set_xlabel("Epoch")
        ax[idx].set_ylabel(m)
    ax[-1].legend()
    fig.suptitle(f"{ds_name} Validation Metrics\nSWA | CWA | SCHM")
    fname = os.path.join(working_dir, f"{ds_name}_metrics_curves.png")
    plt.savefig(fname)
    plt.close(fig)
except Exception as e:
    print(f"Error creating metrics figure: {e}")
    plt.close()

# -------------- print final SCHM --------------------------
for lr_key, d in lr_dict.items():
    schm = d[ds_name]["metrics"]["SCHM"][-1] if d[ds_name]["metrics"]["SCHM"] else None
    print(
        f"Final SCHM for lr={lr_label(lr_key)}: {schm:.3f}"
        if schm is not None
        else f"No SCHM for lr={lr_label(lr_key)}"
    )
