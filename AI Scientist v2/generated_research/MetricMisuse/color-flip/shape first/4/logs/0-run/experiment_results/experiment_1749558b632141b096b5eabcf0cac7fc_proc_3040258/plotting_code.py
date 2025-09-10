import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load the stored experiment dict
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

models = ["UNI_LSTM", "BI_LSTM"]
dataset = "SPR_BENCH"

# 1) Final HWA vs epochs (both models)
try:
    plt.figure()
    for m in models:
        epochs = sorted(int(k) for k in experiment_data[m][dataset].keys())
        hwas = [
            experiment_data[m][dataset][str(e)]["metrics"]["val"][-1] for e in epochs
        ]
        plt.plot(epochs, hwas, marker="o", label=m)
    plt.title("Final Harmonic Weighted Accuracy vs Epochs (SPR_BENCH)")
    plt.xlabel("Training epochs")
    plt.ylabel("HWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_HWA_vs_epochs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA_vs_epochs: {e}")
    plt.close()


# helper to fetch the longest run record for a model
def longest_run(model):
    ep = max(int(k) for k in experiment_data[model][dataset].keys())
    return ep, experiment_data[model][dataset][str(ep)]


# 2) & 3) Loss curves for longest runs
for m in models:
    try:
        ep, rec = longest_run(m)
        plt.figure()
        plt.plot(range(1, ep + 1), rec["losses"]["train"], label="Train")
        plt.plot(range(1, ep + 1), rec["losses"]["val"], label="Validation")
        plt.title(f"{m} Loss Curves ({ep} Epochs) - SPR_BENCH")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_{m}_loss_curves_{ep}ep.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {m}: {e}")
        plt.close()

# 4) Per-epoch HWA for longest runs (both models)
try:
    plt.figure()
    for m in models:
        ep, rec = longest_run(m)
        plt.plot(range(1, ep + 1), rec["metrics"]["val"], label=f"{m} ({ep}ep)")
    plt.title("Per-Epoch HWA (Longest Runs) - SPR_BENCH")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_HWA_curves_longest_runs.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating per-epoch HWA curves: {e}")
    plt.close()
