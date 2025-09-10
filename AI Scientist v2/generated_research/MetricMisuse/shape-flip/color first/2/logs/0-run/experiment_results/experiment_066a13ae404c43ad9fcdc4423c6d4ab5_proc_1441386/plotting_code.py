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

poolings = list(experiment_data.get("pooling_type", {}).keys())
epochs_dict = {}

# Pre-extract data for ease
for p in poolings:
    log = experiment_data["pooling_type"][p]["SPR_BENCH"]
    epochs_dict[p] = {
        "train_loss": [v for _, v in log["losses"]["train"]],
        "val_loss": [v for _, v in log["losses"]["val"]],
        "dwa": [v for _, v in log["metrics"]["val"]],
    }

# 1) Loss curves
try:
    plt.figure(figsize=(8, 5))
    for p in poolings:
        ep = np.arange(1, len(epochs_dict[p]["train_loss"]) + 1)
        plt.plot(ep, epochs_dict[p]["train_loss"], linestyle="--", label=f"{p}-train")
        plt.plot(ep, epochs_dict[p]["val_loss"], linestyle="-", label=f"{p}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves - SPR_BENCH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# 2) DWA curves
try:
    plt.figure(figsize=(8, 5))
    for p in poolings:
        ep = np.arange(1, len(epochs_dict[p]["dwa"]) + 1)
        plt.plot(ep, epochs_dict[p]["dwa"], label=p)
    plt.xlabel("Epoch")
    plt.ylabel("Dual Weighted Accuracy")
    plt.title("Validation Dual Weighted Accuracy Curves - SPR_BENCH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_dwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating DWA curves plot: {e}")
    plt.close()

# 3) Final DWA bar chart
try:
    plt.figure(figsize=(6, 4))
    final_scores = [epochs_dict[p]["dwa"][-1] for p in poolings]
    plt.bar(poolings, final_scores, color="skyblue")
    plt.ylabel("Final Dual Weighted Accuracy")
    plt.title("Final DWA by Pooling Type - SPR_BENCH")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_dwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final DWA bar plot: {e}")
    plt.close()
