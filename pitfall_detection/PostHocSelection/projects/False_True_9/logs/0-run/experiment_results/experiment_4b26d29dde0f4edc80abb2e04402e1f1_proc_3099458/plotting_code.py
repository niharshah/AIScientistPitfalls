import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load saved experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tags = sorted(
    experiment_data.keys(), key=lambda x: int(x.split("_")[-1])
)  # emb_dim_32 -> 32

# Figure 1: Train vs Val loss across emb dims
try:
    plt.figure()
    for tag in tags:
        train = experiment_data[tag]["SPR_BENCH"]["losses"]["train"]
        val = experiment_data[tag]["SPR_BENCH"]["losses"]["val"]
        epochs_t, loss_t = zip(*train)
        epochs_v, loss_v = zip(*val)
        plt.plot(epochs_t, loss_t, "--", label=f"{tag}-train")
        plt.plot(epochs_v, loss_v, "-", label=f"{tag}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Figure 2: Validation metrics (HWA) over epochs
try:
    plt.figure()
    for tag in tags:
        metrics = experiment_data[tag]["SPR_BENCH"]["metrics"]["val"]
        epochs, swa, cwa, hwa = zip(*metrics)
        plt.plot(epochs, hwa, label=f"{tag}-HWA")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: Validation HWA across Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_HWA_over_epochs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA over epochs plot: {e}")
    plt.close()

# Figure 3: Final-epoch HWA bar chart
try:
    plt.figure()
    hw_vals = []
    for tag in tags:
        metrics = experiment_data[tag]["SPR_BENCH"]["metrics"]["val"]
        hw = metrics[-1][-1]  # last epoch HWA
        hw_vals.append(hw)
    plt.bar(tags, hw_vals)
    plt.ylabel("Final HWA")
    plt.title("SPR_BENCH: Final Harmonic Weighted Accuracy by Embedding Dimension")
    plt.xticks(rotation=45)
    fname = os.path.join(working_dir, "SPR_BENCH_final_HWA_bar.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    # Print summary
    print("Final HWA per embedding dimension:")
    for tag, val in zip(tags, hw_vals):
        print(f"{tag}: {val:.4f}")
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()
