import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = exp["batch_size_tuning"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# helper to keep consistent colors
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
bs_list = sorted(runs.keys(), key=lambda x: int(x.split("_")[1]))  # ['bs_64', ...]
color_map = {bs: colors[i % len(colors)] for i, bs in enumerate(bs_list)}

# ---------- 1. training loss curves ----------
try:
    plt.figure()
    for bs in bs_list:
        tr = runs[bs]["losses"]["train"]
        plt.plot(tr, label=f"{bs} train", color=color_map[bs], linestyle="-")
    plt.title("Training Loss vs Epoch (SPR Bench)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_training_loss_batch_size_tuning.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# ---------- 2. validation loss curves ----------
try:
    plt.figure()
    for bs in bs_list:
        val = runs[bs]["losses"]["val"]
        plt.plot(val, label=f"{bs} val", color=color_map[bs], linestyle="--")
    plt.title("Validation Loss vs Epoch (SPR Bench)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_validation_loss_batch_size_tuning.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()

# ---------- 3. HWA per epoch ----------
try:
    plt.figure()
    for bs in bs_list:
        hwa = [tpl[2] for tpl in runs[bs]["metrics"]["val"]]
        plt.plot(hwa, label=f"{bs}", color=color_map[bs])
    plt.title("HWA (Validation) vs Epoch (SPR Bench)")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Acc.")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_hwa_curves_batch_size_tuning.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ---------- 4. final test metric comparison ----------
try:
    swa_vals, cwa_vals, hwa_vals = [], [], []
    for bs in bs_list:
        swa, cwa, hwa = runs[bs]["metrics"]["test"]
        swa_vals.append(swa)
        cwa_vals.append(cwa)
        hwa_vals.append(hwa)

    x = np.arange(len(bs_list))
    width = 0.25
    plt.figure(figsize=(8, 4))
    plt.bar(x - width, swa_vals, width, label="SWA")
    plt.bar(x, cwa_vals, width, label="CWA")
    plt.bar(x + width, hwa_vals, width, label="HWA")
    plt.xticks(x, bs_list)
    plt.title("Final Test Accuracies by Batch Size (SPR Bench)")
    plt.ylabel("Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_test_metrics_batch_size_tuning.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar plot: {e}")
    plt.close()

# ---------- print summary ----------
if runs:
    print("\nFinal Test HWA by Batch Size")
    for bs in bs_list:
        hwa = runs[bs]["metrics"]["test"][2]
        print(f"{bs}: {hwa:.4f}")
