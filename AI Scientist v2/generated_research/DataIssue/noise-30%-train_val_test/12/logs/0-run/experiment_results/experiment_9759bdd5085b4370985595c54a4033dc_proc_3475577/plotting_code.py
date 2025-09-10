import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ LOAD DATA ------------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Quick helper to retrieve safely
def get_bs_dict(data_dict):
    try:
        return data_dict["frozen_embedding"]["SPR_BENCH"]["batch_size"]
    except KeyError:
        return {}


bs_dict = get_bs_dict(experiment_data)
batch_sizes = sorted(bs_dict.keys())

# ------------------ PLOT FIGURES ------------------
# 1) Combined Loss Curves -------------------------------------------------
try:
    plt.figure()
    for bs in batch_sizes:
        epochs = bs_dict[bs]["epochs"]
        tr_loss = bs_dict[bs]["losses"]["train"]
        val_loss = bs_dict[bs]["losses"]["val"]
        plt.plot(epochs, tr_loss, label=f"train_loss_bs{bs}")
        plt.plot(epochs, val_loss, linestyle="--", label=f"val_loss_bs{bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        "SPR_BENCH Frozen-Embedding\nLeft: Train Loss, Right: Val Loss (all batch sizes)"
    )
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "spr_bench_frozen_loss_all_bs.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating combined loss plot: {e}")
    plt.close()

# 2) Combined Validation F1 Curves ---------------------------------------
try:
    plt.figure()
    for bs in batch_sizes:
        epochs = bs_dict[bs]["epochs"]
        val_f1 = bs_dict[bs]["metrics"]["val_f1"]
        plt.plot(epochs, val_f1, label=f"val_f1_bs{bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Frozen-Embedding\nValidation Macro-F1 vs Epoch")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "spr_bench_frozen_valF1_all_bs.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating combined val-F1 plot: {e}")
    plt.close()

# 3) Final Epoch Val-F1 Bar Chart ----------------------------------------
try:
    plt.figure()
    final_f1s = [bs_dict[bs]["metrics"]["val_f1"][-1] for bs in batch_sizes]
    plt.bar(range(len(batch_sizes)), final_f1s, tick_label=batch_sizes)
    plt.xlabel("Batch Size")
    plt.ylabel("Final Epoch Macro-F1")
    plt.title("SPR_BENCH Frozen-Embedding\nFinal Validation Macro-F1 by Batch Size")
    fname = os.path.join(working_dir, "spr_bench_frozen_final_valF1_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating final val-F1 bar chart: {e}")
    plt.close()

print("Plot generation complete. Files saved to 'working/'.")
