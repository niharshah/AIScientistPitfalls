import matplotlib.pyplot as plt
import numpy as np
import os

# set up paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
experiment_path = os.path.join(working_dir, "experiment_data.npy")
try:
    experiment_data = np.load(experiment_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# helper to fetch nested dict
bench_key = ("batch_size_tuning", "SPR_BENCH")
exp = experiment_data.get(bench_key[0], {}).get(bench_key[1], {})

# print best val F1 for every batch size
for bs, d in exp.items():
    val_f1s = [e["macro_f1"] for e in d["metrics"]["val"]]
    if val_f1s:
        print(f"Batch size {bs}: best val macro-F1 = {max(val_f1s):.4f}")

# COLORS for consistency
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# ------------- Plot 1: Macro-F1 curves -------------
try:
    plt.figure()
    for idx, (bs, d) in enumerate(sorted(exp.items())):
        epochs = [e["epoch"] for e in d["metrics"]["train"]]
        tr_f1 = [e["macro_f1"] for e in d["metrics"]["train"]]
        val_f1 = [e["macro_f1"] for e in d["metrics"]["val"]]
        c = COLORS[idx % len(COLORS)]
        plt.plot(epochs, tr_f1, color=c, label=f"train_bs{bs}")
        plt.plot(epochs, val_f1, color=c, linestyle="--", label=f"val_bs{bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 vs Epoch (Batch-Size Tuning)")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_macro_f1_batch_size_tuning.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating Macro-F1 plot: {e}")
    plt.close()

# ------------- Plot 2: Loss curves -------------
try:
    plt.figure()
    for idx, (bs, d) in enumerate(sorted(exp.items())):
        epochs = [e["epoch"] for e in d["losses"]["train"]]
        tr_loss = [e["loss"] for e in d["losses"]["train"]]
        val_loss = [e["loss"] for e in d["losses"]["val"]]
        c = COLORS[idx % len(COLORS)]
        plt.plot(epochs, tr_loss, color=c, label=f"train_bs{bs}")
        plt.plot(epochs, val_loss, color=c, linestyle="--", label=f"val_bs{bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss vs Epoch (Batch-Size Tuning)")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_batch_size_tuning.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating Loss plot: {e}")
    plt.close()
