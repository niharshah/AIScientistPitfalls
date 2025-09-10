import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data["gradient_clip_max_norm"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# print and collect final F1s
final_f1s, tags = [], []
for tag, run in runs.items():
    f1 = run["metrics"]["val_f1"][-1] if run["metrics"]["val_f1"] else np.nan
    print(f"Final Val Macro-F1 ({tag}): {f1:.4f}")
    final_f1s.append(f1)
    tags.append(tag)

# ----- PLOTS -----
# 1-3) Loss curves for first three runs
for i, tag in enumerate(tags[:3]):
    try:
        data = runs[tag]
        plt.figure()
        plt.plot(data["epochs"], data["losses"]["train"], label="Train Loss")
        plt.plot(data["epochs"], data["losses"]["val"], label="Val Loss")
        plt.title(f"SPR-BENCH Loss Curve ({tag})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"loss_curve_{tag}.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {tag}: {e}")
        plt.close()

# 4) Combined validation F1 curves
try:
    plt.figure()
    for tag in tags:
        plt.plot(runs[tag]["epochs"], runs[tag]["metrics"]["val_f1"], label=tag)
    plt.title("SPR-BENCH Validation Macro-F1 vs Epoch (All Runs)")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    fname = os.path.join(working_dir, "val_f1_all_runs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating combined F1 plot: {e}")
    plt.close()

# 5) Bar chart of final F1
try:
    plt.figure()
    plt.bar(tags, final_f1s)
    plt.title("Final Epoch Macro-F1 by Gradient Clipping (SPR-BENCH)")
    plt.ylabel("Macro-F1")
    plt.xticks(rotation=45, ha="right")
    fname = os.path.join(working_dir, "final_f1_bar.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final F1 bar plot: {e}")
    plt.close()
