import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tags = sorted(experiment_data.keys())
epochs = range(
    1, 1 + max(len(experiment_data[t]["SPR_BENCH"]["losses"]["train"]) for t in tags)
)


# helper to fetch list safely
def get_list(tag, key_path):
    d = experiment_data[tag]["SPR_BENCH"]
    for k in key_path:
        d = d[k]
    return d


# ---------- 1. training loss ----------
try:
    plt.figure()
    for tag in tags:
        plt.plot(epochs, get_list(tag, ["losses", "train"]), label=tag)
    plt.title("SPR_BENCH: Training Loss vs Epoch (Dropout Ablation)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_loss_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# ---------- 2. validation loss ----------
try:
    plt.figure()
    for tag in tags:
        plt.plot(epochs, get_list(tag, ["losses", "val"]), label=tag)
    plt.title("SPR_BENCH: Validation Loss vs Epoch (Dropout Ablation)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_loss_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()

# ---------- 3. validation HWA ----------
try:
    plt.figure()
    for tag in tags:
        hwa = [m["hwa"] for m in get_list(tag, ["metrics", "val"])]
        plt.plot(epochs, hwa, label=tag)
    plt.title("SPR_BENCH: Validation HWA vs Epoch (Dropout Ablation)")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_HWA_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ---------- 4. final HWA bar chart ----------
try:
    plt.figure()
    final_hwa = [get_list(tag, ["metrics", "val"])[-1]["hwa"] for tag in tags]
    plt.bar(tags, final_hwa, color="skyblue")
    plt.title("SPR_BENCH: Final-Epoch HWA by Dropout")
    plt.ylabel("HWA")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_HWA_dropout.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

# ---------- print summary ----------
for tag, hwa in zip(tags, final_hwa if "final_hwa" in locals() else []):
    print(f"{tag}: final validation HWA = {hwa:.4f}")
