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

# parameters
dataset = "SPR_BENCH"
epoch_options = ["5", "10", "20", "30"]
variants = {"baseline": "Baseline", "frozen_emb": "Frozen-Emb"}

# plotting
for e in epoch_options[:5]:  # ensure at most 5 figs
    try:
        plt.figure(figsize=(10, 4))
        # Left subplot: loss curves
        ax1 = plt.subplot(1, 2, 1)
        for var_key, var_name in variants.items():
            rec = experiment_data.get(var_key, {}).get(dataset, {}).get(e, {})
            tr = rec.get("losses", {}).get("train", [])
            va = rec.get("losses", {}).get("val", [])
            if tr:
                ax1.plot(range(1, len(tr) + 1), tr, label=f"{var_name} Train")
            if va:
                ax1.plot(range(1, len(va) + 1), va, "--", label=f"{var_name} Val")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training vs Validation Loss")
        ax1.legend()

        # Right subplot: HWA curves
        ax2 = plt.subplot(1, 2, 2)
        for var_key, var_name in variants.items():
            rec = experiment_data.get(var_key, {}).get(dataset, {}).get(e, {})
            hwa = rec.get("metrics", {}).get("val", [])
            if hwa:
                ax2.plot(range(1, len(hwa) + 1), hwa, label=f"{var_name} HWA")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("HWA")
        ax2.set_title("Harmonic Weighted Accuracy")
        ax2.legend()

        plt.suptitle(
            f"{dataset} Epochs={e}\nLeft: Train/Val Loss, Right: HWA (SPR_BENCH)"
        )
        fname = f"{dataset}_e{e}_curves.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as err:
        print(f"Error creating plot for epoch {e}: {err}")
        plt.close()
