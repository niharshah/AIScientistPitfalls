import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    raise RuntimeError(f"Could not load experiment_data.npy: {e}")

batch_dict = experiment_data.get("batch_size", {})
dataset_name = "SPR_synthetic"  # only dataset present in supplied experiment

# 1) Loss curves --------------------------------------------------------------
try:
    plt.figure()
    for tag, bd in batch_dict.items():
        epochs = bd["epochs"]
        plt.plot(epochs, bd["losses"]["train"], "--", label=f"train bs={tag}")
        plt.plot(epochs, bd["losses"]["val"], "-", label=f"val   bs={tag}")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(f"{dataset_name}: Training vs Validation Loss\nBatch-size comparison")
    plt.legend(fontsize=7)
    save_path = os.path.join(working_dir, f"{dataset_name}_loss_curves.png")
    plt.savefig(save_path, dpi=150)
    print("Saved:", save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) SDWA metric curves --------------------------------------------------------
try:
    plt.figure()
    for tag, bd in batch_dict.items():
        epochs = bd["epochs"]
        plt.plot(epochs, bd["metrics"]["train"], "--", label=f"train bs={tag}")
        plt.plot(epochs, bd["metrics"]["val"], "-", label=f"val   bs={tag}")
    plt.xlabel("epoch")
    plt.ylabel("SDWA score")
    plt.title(f"{dataset_name}: Training vs Validation SDWA\nBatch-size comparison")
    plt.legend(fontsize=7)
    save_path = os.path.join(working_dir, f"{dataset_name}_sdwa_curves.png")
    plt.savefig(save_path, dpi=150)
    print("Saved:", save_path)
    plt.close()
except Exception as e:
    print(f"Error creating SDWA plot: {e}")
    plt.close()

# 3) Final test SDWA bar plot --------------------------------------------------
try:
    plt.figure()
    tags, scores = [], []
    for tag, bd in batch_dict.items():
        tags.append(str(tag))
        scores.append(bd.get("test_SDWA", np.nan))
    plt.bar(tags, scores, color="skyblue")
    plt.xlabel("batch size")
    plt.ylabel("Test SDWA")
    plt.title(f"{dataset_name}: Final Test SDWA per Batch Size")
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    save_path = os.path.join(working_dir, f"{dataset_name}_test_sdwa_bar.png")
    plt.savefig(save_path, dpi=150)
    print("Saved:", save_path)
    plt.close()
except Exception as e:
    print(f"Error creating Test SDWA bar plot: {e}")
    plt.close()
