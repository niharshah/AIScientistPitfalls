import matplotlib.pyplot as plt
import numpy as np
import os

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_bs_dict = experiment_data.get("batch_size", {}).get("SPR_BENCH", {})

# Collect keys and ensure consistent ordering
bs_keys = sorted(spr_bs_dict.keys(), key=lambda x: int(x.split("_")[-1]))

# 1) Loss curves ----------------------------------------------------
try:
    plt.figure(figsize=(8, 5))
    for k in bs_keys:
        losses = spr_bs_dict[k]["losses"]
        plt.plot(losses["train"], label=f"{k}-train")
        plt.plot(losses["val"], label=f"{k}-val", linestyle="--")
    plt.title("SPR_BENCH: Training vs Validation Loss (GRU baseline)")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) MCC curves -----------------------------------------------------
try:
    plt.figure(figsize=(8, 5))
    for k in bs_keys:
        mccs = spr_bs_dict[k]["metrics"]
        plt.plot(mccs["train"], label=f"{k}-train")
        plt.plot(mccs["val"], label=f"{k}-val", linestyle="--")
    plt.title("SPR_BENCH: Training vs Validation MCC")
    plt.xlabel("Epoch")
    plt.ylabel("Matthews Correlation Coefficient")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_MCC_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating MCC curve plot: {e}")
    plt.close()

# 3) Test MCC bar chart --------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    test_mccs = [spr_bs_dict[k].get("test_mcc", np.nan) for k in bs_keys]
    plt.bar(
        range(len(bs_keys)), test_mccs, tick_label=[k.split("_")[-1] for k in bs_keys]
    )
    plt.title("SPR_BENCH: Test MCC by Batch Size")
    plt.ylabel("Matthews Correlation Coefficient")
    plt.xlabel("Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_test_MCC_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Test MCC bar plot: {e}")
    plt.close()

# Print numeric results --------------------------------------------
for k in bs_keys:
    print(f"{k}: Test MCC = {spr_bs_dict[k].get('test_mcc', 'NA')}")
