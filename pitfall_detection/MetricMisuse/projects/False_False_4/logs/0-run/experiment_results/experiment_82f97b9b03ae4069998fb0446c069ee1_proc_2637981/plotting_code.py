import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment results ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

datasets = list(experiment_data.keys())  # top-level experiment names

# ---------- per-dataset plots ----------
for ds in datasets:
    data = experiment_data.get(ds, {})
    losses = data.get("losses", {})
    metrics = data.get("metrics", {})
    test_metrics = data.get("test_metrics", {})
    # -------- Loss curves --------
    try:
        plt.figure()
        if "train" in losses:
            plt.plot(losses["train"], label="train")
        if "dev" in losses:
            plt.plot(losses["dev"], label="dev", linestyle="--")
        if plt.gca().has_data():
            plt.title(f"{ds} Loss Curves\nTrain vs Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_loss_curves.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {ds}: {e}")
        plt.close()

    # -------- Accuracy curves --------
    try:
        plt.figure()
        if "train_acc" in metrics:
            plt.plot(metrics["train_acc"], label="train")
        if "dev_acc" in metrics:
            plt.plot(metrics["dev_acc"], label="dev", linestyle="--")
        if plt.gca().has_data():
            plt.title(f"{ds} Accuracy Curves\nTrain vs Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_accuracy_curves.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curves for {ds}: {e}")
        plt.close()

    # -------- Dev-level auxiliary metric (e.g. SWA) --------
    try:
        if "dev_swa" in metrics:
            plt.figure()
            plt.plot(metrics["dev_swa"], label="dev_swa")
            plt.title(f"{ds} Validation Shape-Weighted Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_dev_swa.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating dev_swa plot for {ds}: {e}")
        plt.close()

    # -------- Final test metrics bar plot --------
    try:
        if test_metrics:
            names, vals = zip(*test_metrics.items())
            x = np.arange(len(names))
            plt.figure()
            plt.bar(x, vals, color="skyblue")
            plt.xticks(x, names, rotation=45, ha="right")
            plt.title(f"{ds} Final Test Metrics")
            plt.ylabel("Score")
            fname = os.path.join(working_dir, f"{ds}_test_metrics.png")
            plt.tight_layout()
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics bar plot for {ds}: {e}")
        plt.close()

# ---------- cross-dataset comparison: test accuracy ----------
try:
    accs = [experiment_data[d].get("test_metrics", {}).get("acc") for d in datasets]
    valid = [(d, a) for d, a in zip(datasets, accs) if a is not None]
    if valid:
        labels, values = zip(*valid)
        x = np.arange(len(labels))
        plt.figure()
        plt.bar(x, values, color="lightgreen")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.title("Final Test Accuracy Comparison Across Datasets")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "cross_dataset_test_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating cross-dataset accuracy comparison: {e}")
    plt.close()

print(f"Plots saved for datasets: {', '.join(datasets)}")
