import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment results --------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = exp["batch_size"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = {}

# helper to fetch colorful linestyle cycling
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
ls = ["-", "--"]

# -------- 1) F1 curve --------
try:
    plt.figure()
    for i, (bs, rec) in enumerate(
        sorted(data.items(), key=lambda x: int(x[0].split("_")[-1]))
    ):
        epochs = rec["epochs"]
        plt.plot(
            epochs,
            rec["metrics"]["train"],
            color=colors[i % len(colors)],
            linestyle=ls[0],
            label=f'bs{bs.split("_")[-1]}-train',
        )
        plt.plot(
            epochs,
            rec["metrics"]["val"],
            color=colors[i % len(colors)],
            linestyle=ls[1],
            label=f'bs{bs.split("_")[-1]}-val',
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH: Training vs Validation Macro-F1")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# -------- 2) Loss curve --------
try:
    plt.figure()
    for i, (bs, rec) in enumerate(
        sorted(data.items(), key=lambda x: int(x[0].split("_")[-1]))
    ):
        epochs = rec["epochs"]
        plt.plot(
            epochs,
            rec["losses"]["train"],
            color=colors[i % len(colors)],
            linestyle=ls[0],
            label=f'bs{bs.split("_")[-1]}-train',
        )
        plt.plot(
            epochs,
            rec["losses"]["val"],
            color=colors[i % len(colors)],
            linestyle=ls[1],
            label=f'bs{bs.split("_")[-1]}-val',
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------- 3) Test F1 bar chart --------
try:
    plt.figure()
    bs_list, test_f1s = [], []
    for bs, rec in sorted(data.items(), key=lambda x: int(x[0].split("_")[-1])):
        bs_list.append(bs.split("_")[-1])
        test_f1s.append(rec["test_f1"])
    plt.bar(bs_list, test_f1s, color="tab:cyan")
    plt.xlabel("Batch Size")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH: Test Macro-F1 by Batch Size")
    for idx, val in enumerate(test_f1s):
        plt.text(idx, val + 0.01, f"{val:.2f}", ha="center")
    fname = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test F1 bar chart: {e}")
    plt.close()
