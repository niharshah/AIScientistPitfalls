import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir path
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment results --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# quick exit if data missing
if not experiment_data:
    quit()

cfgs = experiment_data["num_epochs"]["SPR_BENCH"]["configurations"]
cfg_names = sorted(cfgs.keys(), key=lambda x: int(x.split("_")[-1]))

# print summary table
print("\nConfig\tBest Dev F1\tTest F1")
for n in cfg_names:
    r = cfgs[n]
    print(f"{n}\t{r['best_dev_f1']:.4f}\t{r['test_f1']:.4f}")

best_cfg = max(cfg_names, key=lambda c: cfgs[c]["best_dev_f1"])
print(f"\nBest configuration on dev set: {best_cfg}")

# -------- 1) bar chart of best dev F1 --------
try:
    plt.figure()
    best_f1s = [cfgs[n]["best_dev_f1"] for n in cfg_names]
    plt.bar(cfg_names, best_f1s, color="skyblue")
    plt.ylabel("Best Dev F1")
    plt.title("SPR_BENCH – Best Dev-set F1 per Epoch Setting")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_best_dev_f1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()


# -------- helper for multi-line curves --------
def multilines(metric_key, ylabel, fname_suffix):
    plt.figure()
    for n in cfg_names:
        epochs = cfgs[n]["epochs"]
        tr = cfgs[n][metric_key]["train"]
        val = cfgs[n][metric_key]["val"]
        plt.plot(epochs, tr, "--", label=f"{n}-train")
        plt.plot(epochs, val, "-", label=f"{n}-val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"SPR_BENCH – {ylabel} Curves for Different Epoch Settings")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_{fname_suffix}_curves.png"))
    plt.close()


# -------- 2) F1 curves --------
try:
    multilines("metrics", "F1 Score", "F1")
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# -------- 3) Loss curves --------
try:
    multilines("losses", "Cross-Entropy Loss", "loss")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()
