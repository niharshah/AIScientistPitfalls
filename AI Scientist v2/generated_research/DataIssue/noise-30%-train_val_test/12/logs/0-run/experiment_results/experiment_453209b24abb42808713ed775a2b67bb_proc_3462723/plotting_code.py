import matplotlib.pyplot as plt
import numpy as np
import os

# ---- paths ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load data ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

batch_dict = experiment_data.get("batch_size", {})

# ---- pre–aggregate ----
epochs_dict, tr_loss_dict, val_loss_dict, val_f1_dict, best_f1 = {}, {}, {}, {}, {}
for bs, stats in batch_dict.items():
    epochs_dict[bs] = np.array(stats["epochs"])
    tr_loss_dict[bs] = np.array(stats["losses"]["train"])
    val_loss_dict[bs] = np.array(stats["losses"]["val"])
    val_f1_dict[bs] = np.array(stats["metrics"]["val_f1"])
    best_f1[bs] = val_f1_dict[bs].max()

# ---- 1: training loss curves ----
try:
    plt.figure()
    for bs, losses in tr_loss_dict.items():
        plt.plot(epochs_dict[bs], losses, label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Training Loss vs Epoch")
    plt.legend()
    fname = os.path.join(working_dir, "spr_train_loss_all_bs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating training-loss plot: {e}")
    plt.close()

# ---- 2: validation loss curves ----
try:
    plt.figure()
    for bs, losses in val_loss_dict.items():
        plt.plot(epochs_dict[bs], losses, label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Validation Loss vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_val_loss_all_bs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation-loss plot: {e}")
    plt.close()

# ---- 3: validation macro-F1 curves ----
try:
    plt.figure()
    for bs, f1s in val_f1_dict.items():
        plt.plot(epochs_dict[bs], f1s, label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH Validation Macro-F1 vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_val_f1_all_bs.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation-F1 plot: {e}")
    plt.close()

# ---- 4: best macro-F1 per batch size ----
try:
    plt.figure()
    bs_vals, f1_vals = zip(*sorted(best_f1.items()))
    plt.bar(range(len(bs_vals)), f1_vals, tick_label=list(bs_vals))
    plt.xlabel("Batch Size")
    plt.ylabel("Best Macro F1")
    plt.title("SPR_BENCH Best Validation Macro-F1 by Batch Size")
    plt.savefig(os.path.join(working_dir, "spr_best_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating best-F1 bar plot: {e}")
    plt.close()

# ---- numeric summary ----
for bs in sorted(best_f1):
    print(f"Batch size {bs:>3}: best val Macro-F1 = {best_f1[bs]:.4f}")
