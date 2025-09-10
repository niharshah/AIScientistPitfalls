import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("WEIGHT_DECAY", {}).get("SPR_BENCH", {})

# Collect final test MCCs for printing later
test_mcc_dict = {}

# ------------------------------------------------------------------ plot 1: loss curves
try:
    plt.figure()
    for key, run in runs.items():
        epochs = run["epochs"]
        plt.plot(epochs, run["losses"]["train"], linestyle="--", label=f"{key}-train")
        plt.plot(epochs, run["losses"]["val"], linestyle="-", label=f"{key}-val")
    plt.title("Training vs Validation Loss\nDataset: SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "spr_bench_loss_weight_decay.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------------------------------------------------------------ plot 2: validation MCC
try:
    plt.figure()
    for key, run in runs.items():
        epochs = run["epochs"]
        plt.plot(epochs, run["metrics"]["val_MCC"], label=key)
    plt.title("Validation MCC across Epochs\nDataset: SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "spr_bench_val_mcc_weight_decay.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation MCC plot: {e}")
    plt.close()

# ------------------------------------------------------------------ plot 3: final test MCC bar chart
try:
    wd_labels, test_mccs = [], []
    for key, run in runs.items():
        wd_labels.append(key.replace("wd_", "wd="))
        mcc = run["metrics"]["test_MCC"]
        test_mccs.append(mcc)
        test_mcc_dict[wd_labels[-1]] = mcc
    plt.figure()
    plt.bar(wd_labels, test_mccs)
    plt.title("Final Test MCC per Weight Decay\nDataset: SPR_BENCH")
    plt.ylabel("MCC")
    plt.xticks(rotation=45, ha="right")
    fname = os.path.join(working_dir, "spr_bench_test_mcc_weight_decay.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test MCC bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------ print evaluation metrics
print("Test MCC per weight decay:", test_mcc_dict)
