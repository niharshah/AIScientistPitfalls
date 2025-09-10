import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch metrics safely
def get_metric(tag, key):
    return experiment_data["gradient_clipping_max_norm"]["SPR_BENCH"][tag]["metrics"][
        key
    ]


tags = list(
    experiment_data.get("gradient_clipping_max_norm", {}).get("SPR_BENCH", {}).keys()
)
epochs = (
    np.arange(1, len(get_metric(tags[0], "train_loss")) + 1) if tags else np.array([])
)

# 1) loss curves
try:
    plt.figure()
    for t in tags:
        plt.plot(epochs, get_metric(t, "train_loss"), label=f"{t}-train")
        plt.plot(epochs, get_metric(t, "val_loss"), "--", label=f"{t}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH - Training & Validation Loss\n(Gradient Clipping Max Norm)")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) accuracy-type curves
try:
    plt.figure()
    for met, style in zip(["SWA", "CWA", "HWA"], ["-", "--", ":"]):
        for t in tags:
            plt.plot(epochs, get_metric(t, met), linestyle=style, label=f"{t}-{met}")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("SPR_BENCH - SWA/CWA/HWA over Epochs\n(Gradient Clipping Max Norm)")
    plt.legend(fontsize=6, ncol=3)
    fname = os.path.join(working_dir, "SPR_BENCH_weighted_accuracy_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 3) final-epoch HWA summary
try:
    plt.figure()
    hwa_final = [get_metric(t, "HWA")[-1] for t in tags]
    plt.bar(range(len(tags)), hwa_final, color="skyblue")
    plt.xticks(range(len(tags)), [t.replace("clip_", "") for t in tags])
    plt.xlabel("Gradient Clipping Max Norm")
    plt.ylabel("Final-Epoch HWA")
    plt.title("SPR_BENCH - Final HWA vs Gradient Clipping")
    fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating HWA bar chart: {e}")
    plt.close()

print("Finished plotting, files saved to:", working_dir)
