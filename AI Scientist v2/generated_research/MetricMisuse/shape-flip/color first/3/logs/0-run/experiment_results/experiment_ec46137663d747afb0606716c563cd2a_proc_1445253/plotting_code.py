import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("weight_decay", {})
tags = list(runs.keys())
num_epochs = max(len(runs[t]["losses"]["train"]) for t in tags) if tags else 0
e = np.arange(1, num_epochs + 1)

# ------------------------------------------------------------------
# 1) Combined loss curves
try:
    plt.figure(figsize=(6, 4))
    for t in tags:
        plt.plot(e, runs[t]["losses"]["train"], label=f"{t} Train")
        plt.plot(e, runs[t]["losses"]["val"], linestyle="--", label=f"{t} Val")
    plt.title(
        "SPR_BENCH Training vs Validation Loss\n(Left: Train, Right: Val curves across weight_decay)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(fontsize=6)
    plt.tight_layout()
    fpath = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fpath)
    plt.close()
    print("Saved", fpath)
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Combined BWA curves
try:
    plt.figure(figsize=(6, 4))
    for t in tags:
        plt.plot(e, runs[t]["metrics"]["train"], label=f"{t} Train")
        plt.plot(e, runs[t]["metrics"]["val"], linestyle="--", label=f"{t} Val")
    plt.title("SPR_BENCH Balanced Weighted Accuracy\n(Left: Train, Right: Val)")
    plt.xlabel("Epoch")
    plt.ylabel("BWA")
    plt.legend(fontsize=6)
    plt.tight_layout()
    fpath = os.path.join(working_dir, "spr_bench_bwa_curves.png")
    plt.savefig(fpath)
    plt.close()
    print("Saved", fpath)
except Exception as e:
    print(f"Error creating BWA curves: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Final test BWA bar chart
try:
    final_bwa = [
        runs[t]["metrics"]["val"][-1] if runs[t]["metrics"]["val"] else 0 for t in tags
    ]
    plt.figure(figsize=(5, 3))
    plt.bar(tags, final_bwa, color="skyblue")
    plt.title("SPR_BENCH Final Dev BWA by Weight Decay")
    plt.xlabel("Weight Decay Tag")
    plt.ylabel("Dev BWA (epoch last)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fpath = os.path.join(working_dir, "spr_bench_final_bwa_bar.png")
    plt.savefig(fpath)
    plt.close()
    print("Saved", fpath)
except Exception as e:
    print(f"Error creating final BWA bar chart: {e}")
    plt.close()
