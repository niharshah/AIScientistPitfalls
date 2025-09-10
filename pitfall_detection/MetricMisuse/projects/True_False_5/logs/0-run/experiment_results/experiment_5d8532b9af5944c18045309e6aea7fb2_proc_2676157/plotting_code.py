import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset_name = "SPR_BENCH"
configs = sorted(experiment_data.keys())  # weight_decay_* strings


# Helper to fetch arrays safely
def get_arr(cfg, section, split):
    return experiment_data[cfg][dataset_name][section].get(split, [])


# -------- Plot 1: Loss curves --------
try:
    plt.figure()
    for cfg in configs:
        train_loss = get_arr(cfg, "losses", "train")
        val_loss = get_arr(cfg, "losses", "val")
        epochs = np.arange(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, "--", label=f"{cfg}-train")
        plt.plot(epochs, val_loss, "-", label=f"{cfg}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nTrain vs. Validation for Different Weight Decays")
    plt.legend()
    out_f = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(out_f, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# -------- Plot 2: Validation RCWA --------
try:
    plt.figure()
    for cfg in configs:
        val_rcwa = get_arr(cfg, "metrics", "val")
        epochs = np.arange(1, len(val_rcwa) + 1)
        plt.plot(epochs, val_rcwa, marker="o", label=cfg)
    plt.xlabel("Epoch")
    plt.ylabel("RCWA")
    plt.title("SPR_BENCH Validation RCWA\nEffect of Weight Decay over Epochs")
    plt.legend()
    out_f = os.path.join(working_dir, "SPR_BENCH_val_RCWA.png")
    plt.savefig(out_f, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating RCWA plot: {e}")
    plt.close()

# -------- Plot 3: Final Test RCWA --------
try:
    plt.figure()
    rcwa_scores = [
        experiment_data[cfg][dataset_name]["test_metrics"]["RCWA"] for cfg in configs
    ]
    plt.bar(range(len(configs)), rcwa_scores, tick_label=configs)
    plt.ylabel("RCWA")
    plt.title("SPR_BENCH Test RCWA\nComparison Across Weight Decays")
    out_f = os.path.join(working_dir, "SPR_BENCH_test_RCWA_bar.png")
    plt.savefig(out_f, dpi=150)
    plt.close()
except Exception as e:
    print(f"Error creating test RCWA bar plot: {e}")
    plt.close()
