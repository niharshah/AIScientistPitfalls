import matplotlib.pyplot as plt
import numpy as np
import os

# --- set up working dir ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load data ---
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# helper to shorten access
runs = experiment_data.get("hidden_dim_tuning", {}).get("SPR_BENCH", {})

# print nothing if empty
if not runs:
    print("No SPR_BENCH data found in experiment_data.npy")
else:
    # ---------------------------------------------------------------
    # 1) Loss curves (train/dev) for all hidden_dim values
    try:
        plt.figure()
        for hd_key, dat in runs.items():
            train_l = dat["losses"]["train"]
            dev_l = dat["losses"]["dev"]
            epochs = range(1, len(train_l) + 1)
            plt.plot(epochs, train_l, "--", label=f"{hd_key} train")
            plt.plot(epochs, dev_l, "-", label=f"{hd_key} dev")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves (Train vs Dev)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------------------------------------------------------------
    # 2) Dev accuracy curves for all hidden_dim values
    try:
        plt.figure()
        for hd_key, dat in runs.items():
            accs = [step["acc"] for step in dat["metrics"]["dev"]]
            epochs = range(1, len(accs) + 1)
            plt.plot(epochs, accs, label=hd_key)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Dev Accuracy Across Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_dev_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating dev accuracy plot: {e}")
        plt.close()

    # ---------------------------------------------------------------
    # 3) Final test metrics bar chart (ACC, SWA, CWA)
    try:
        metrics = ["acc", "swa", "cwa"]
        x = np.arange(len(runs))
        width = 0.25
        plt.figure()
        for i, met in enumerate(metrics):
            vals = [runs[hd]["metrics"]["test"][met] for hd in runs]
            plt.bar(x + i * width, vals, width, label=met.upper())
        plt.xticks(x + width, list(runs.keys()), rotation=45)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Test Metrics by Hidden Dim")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

    # ---------------------------------------------------------------
    # 4) NRGS per hidden_dim
    try:
        plt.figure()
        nrg_vals = [runs[hd]["metrics"]["NRGS"] for hd in runs]
        plt.bar(list(runs.keys()), nrg_vals, color="tab:purple")
        plt.ylabel("NRGS")
        plt.title("SPR_BENCH NRGS by Hidden Dim")
        fname = os.path.join(working_dir, "SPR_BENCH_NRGS.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error creating NRGS plot: {e}")
        plt.close()

    # ---------------------------------------------------------------
    # Console print of final test metrics
    print("\n=== Final Test Metrics ===")
    for hd_key, dat in runs.items():
        t = dat["metrics"]["test"]
        print(
            f"{hd_key}: ACC={t['acc']:.3f}, SWA={t['swa']:.3f}, "
            f"CWA={t['cwa']:.3f}, NRGS={dat['metrics']['NRGS']:.3f}"
        )
