import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

bst_data = experiment_data.get("batch_size_tuning", {})
if not bst_data:
    print("No batch_size_tuning data found.")
else:
    # ------- individual loss curves ----------
    for idx, (tag, exp) in enumerate(bst_data.items()):
        try:
            epochs = list(range(1, len(exp["losses"]["train"]) + 1))
            plt.figure()
            plt.plot(epochs, exp["losses"]["train"], label="Train Loss")
            plt.plot(epochs, exp["losses"]["val"], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"Synthetic Dataset – Loss Curves (batch_size={tag.split('_')[-1]})"
            )
            plt.legend()
            fname = f"synthetic_loss_{tag}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {tag}: {e}")
            plt.close()

    # ------- aggregated HWA comparison --------
    try:
        plt.figure()
        for tag, exp in bst_data.items():
            epochs = list(range(1, len(exp["metrics"]["val"]) + 1))
            hwa = [m["hwa"] for m in exp["metrics"]["val"]]
            plt.plot(epochs, hwa, marker="o", label=f"bs={tag.split('_')[-1]}")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.title("Synthetic Dataset – HWA Comparison Across Batch Sizes")
        plt.legend()
        fname = "synthetic_hwa_batchsize_comparison.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating HWA comparison plot: {e}")
        plt.close()
