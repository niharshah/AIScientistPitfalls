import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset_name = "SPR_BENCH"
runs = experiment_data.get("learning_rate", {}).get(dataset_name, {})
final_cwa = {}

# --------- per-LR plots ----------
for lr_key, rec in runs.items():
    lr_val = lr_key.split("_")[-1]  # e.g. "0.0003"
    # Loss curves
    try:
        plt.figure()
        plt.plot(rec["losses"]["train"], label="Train")
        plt.plot(rec["losses"]["val"], label="Validation")
        plt.title(
            f"{dataset_name} Loss Curves (lr={lr_val})\nLeft: Train, Right: Validation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"{dataset_name}_loss_curve_lr_{lr_val}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {lr_key}: {e}")
        plt.close()

    # Validation CWA curves
    try:
        cwa_vals = rec["metrics"]["val"]
        plt.figure()
        plt.plot(cwa_vals, marker="o")
        plt.title(
            f"{dataset_name} CWA-2D vs Epoch (lr={lr_val})\nSingle curve: Validation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("CWA-2D")
        fname = f"{dataset_name}_cwa_curve_lr_{lr_val}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating CWA plot for {lr_key}: {e}")
        plt.close()

    # store final metric
    if rec["metrics"]["val"]:
        final_cwa[lr_val] = rec["metrics"]["val"][-1]

# --------- bar chart of final CWA across learning rates ----------
try:
    if final_cwa:
        lrs, scores = zip(*sorted(final_cwa.items(), key=lambda x: float(x[0])))
        plt.figure()
        plt.bar(range(len(lrs)), scores, tick_label=lrs)
        plt.title(f"{dataset_name} Final Validation CWA-2D by Learning Rate")
        plt.xlabel("Learning Rate")
        plt.ylabel("Final CWA-2D")
        fname = f"{dataset_name}_final_cwa_bar.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating summary bar plot: {e}")
    plt.close()
