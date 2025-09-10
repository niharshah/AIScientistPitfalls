import matplotlib.pyplot as plt
import numpy as np
import os

# -------- prepare paths & load data --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to safely drill into dict
def get_run(edict, exp, dataset):
    return edict.get(exp, {}).get(dataset, {}) if edict else {}


run = get_run(experiment_data, "remove_token_feature", "SPR_BENCH")
losses = run.get("losses", {})
metrics_val = run.get("metrics", {}).get("val", [])
metrics_test = run.get("metrics", {}).get("test", {})

# -------- plot 1: loss curves --------
try:
    epochs = np.arange(1, len(losses.get("train", [])) + 1)
    plt.figure()
    plt.plot(epochs, losses.get("train", []), label="Train Loss")
    plt.plot(epochs, losses.get("val", []), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Loss Curves (remove_token_feature)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_remove_token_feature.png")
    plt.savefig(fname)
    print("Saved:", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------- plot 2: validation metrics over epochs --------
try:
    if metrics_val:
        epochs = [m["epoch"] for m in metrics_val]
        for key in ["acc", "cwa", "swa", "ccwa"]:
            plt.plot(epochs, [m[key] for m in metrics_val], label=key.upper())
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH – Validation Metrics (remove_token_feature)")
        plt.legend()
        fname = os.path.join(
            working_dir, "SPR_BENCH_val_metrics_remove_token_feature.png"
        )
        plt.savefig(fname)
        print("Saved:", fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# -------- plot 3: test metrics bar chart --------
try:
    if metrics_test:
        keys = ["acc", "cwa", "swa", "ccwa"]
        values = [metrics_test.get(k, 0) for k in keys]
        plt.figure()
        plt.bar(keys, values, color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("SPR_BENCH – Test Metrics (remove_token_feature)")
        fname = os.path.join(
            working_dir, "SPR_BENCH_test_metrics_remove_token_feature.png"
        )
        plt.savefig(fname)
        print("Saved:", fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics plot: {e}")
    plt.close()
