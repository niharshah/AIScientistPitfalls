import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- Load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    abl_data = experiment_data.get("no_padding_mask_ablation", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    abl_data = {}

# Helper to fetch colours consistently
colors = plt.cm.tab10.colors

# ---------------- Plot 1: Loss curves (train & val) ----------------
try:
    plt.figure(figsize=(10, 4))
    # Left subplot: train loss
    plt.subplot(1, 2, 1)
    for i, (bs, stats) in enumerate(sorted(abl_data.items())):
        plt.plot(
            stats["epochs"],
            stats["losses"]["train"],
            label=f"bs={bs}",
            color=colors[i % len(colors)],
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Right subplot: val loss
    plt.subplot(1, 2, 2)
    for i, (bs, stats) in enumerate(sorted(abl_data.items())):
        plt.plot(
            stats["epochs"],
            stats["losses"]["val"],
            label=f"bs={bs}",
            color=colors[i % len(colors)],
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")

    plt.suptitle(
        "No-Pad Mask Ablation – Left: Training Loss, Right: Validation Loss (SPR-Bench)"
    )
    fname = os.path.join(working_dir, "spr_loss_curves_nopad.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# ---------------- Plot 2: Validation F1 curves ----------------
try:
    plt.figure()
    for i, (bs, stats) in enumerate(sorted(abl_data.items())):
        plt.plot(
            stats["epochs"],
            stats["metrics"]["val_f1"],
            label=f"bs={bs}",
            color=colors[i % len(colors)],
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Validation Macro-F1 over Epochs – SPR-Bench")
    plt.legend()
    fname = os.path.join(working_dir, "spr_val_f1_curves_nopad.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves plot: {e}")
    plt.close()

# ---------------- Plot 3: Final epoch F1 bar chart ----------------
try:
    plt.figure()
    bs_vals, final_f1 = [], []
    for bs, stats in sorted(abl_data.items()):
        bs_vals.append(str(bs))
        final_f1.append(stats["metrics"]["val_f1"][-1])
    plt.bar(bs_vals, final_f1, color="skyblue")
    plt.xlabel("Batch Size")
    plt.ylabel("Final-Epoch Macro F1")
    plt.title("Final Validation F1 vs Batch Size – SPR-Bench")
    for i, v in enumerate(final_f1):
        plt.text(i, v + 0.005, f"{v:.2f}", ha="center")
    fname = os.path.join(working_dir, "spr_final_f1_by_bs_nopad.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final F1 bar chart: {e}")
    plt.close()

print(f"Plots saved to {working_dir}")
