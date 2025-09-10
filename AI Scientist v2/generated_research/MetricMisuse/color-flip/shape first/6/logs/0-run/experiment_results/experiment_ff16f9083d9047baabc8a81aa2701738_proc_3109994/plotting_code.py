import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------------------------------------------- load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

datasets = experiment_data.get("multi_synth_rule_diversity", {})  # dict of datasets
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


# ----------------------------------------------------- helper
def get_curve(ex, key):  # key in ['losses','metrics']
    if key == "losses":
        tr = ex["losses"]["train"]
        val = ex["losses"]["val"]
        return list(range(1, len(tr) + 1)), tr, val
    else:  # metrics
        val = [d[key] for d in ex["metrics"]["val"]]
        return list(range(1, len(val) + 1)), val


# ----------------------------------------------------- 1) loss curves
try:
    plt.figure()
    for i, (name, ex) in enumerate(datasets.items()):
        epochs, tr, val = get_curve(ex, "losses")
        c = colors[i % len(colors)]
        plt.plot(epochs, tr, linestyle="--", color=c, label=f"{name}-train")
        plt.plot(epochs, val, linestyle="-", color=c, label=f"{name}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss Curves across Datasets (multi_synth_rule_diversity)")
    plt.legend()
    fname = os.path.join(working_dir, "multi_synth_rule_diversity_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ----------------------------------------------------- metric plots
for metric in ["SCAA", "SWA", "CWA"]:
    try:
        plt.figure()
        for i, (name, ex) in enumerate(datasets.items()):
            epochs, val = get_curve(ex, metric)
            plt.plot(epochs, val, color=colors[i % len(colors)], label=name)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"Validation {metric} over Epochs (multi_synth_rule_diversity)")
        plt.legend()
        fname = os.path.join(
            working_dir, f"multi_synth_rule_diversity_{metric.lower()}_curve.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating {metric} plot: {e}")
        plt.close()

# ----------------------------------------------------- print final metrics
for name, ex in datasets.items():
    last = ex["metrics"]["val"][-1] if ex["metrics"]["val"] else {}
    print(f"{name} final metrics:", last)
