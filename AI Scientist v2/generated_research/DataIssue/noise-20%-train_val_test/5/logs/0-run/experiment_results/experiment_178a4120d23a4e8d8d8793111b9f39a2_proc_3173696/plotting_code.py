import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- LOAD EXPERIMENT DATA ------------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

variants = ["mean_pooling", "cls_token"]
dataset_name = "SPR_BENCH"

# ----------------------- PLOT ACCURACY CURVES ---------------------- #
for v in variants:
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for nhead, res in (
            experiment_data.get(v, {}).get(dataset_name, {}).get("results", {}).items()
        ):
            epochs = range(1, len(res["metrics"]["train_acc"]) + 1)
            axes[0].plot(epochs, res["metrics"]["train_acc"], label=f"nhead={nhead}")
            axes[1].plot(epochs, res["metrics"]["val_acc"], label=f"nhead={nhead}")
        axes[0].set_title("Train Accuracy")
        axes[1].set_title("Validation Accuracy")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()
        fig.suptitle(f"{dataset_name}: {v} (Left: Train, Right: Val)")
        save_path = os.path.join(working_dir, f"{dataset_name}_{v}_accuracy_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error plotting accuracy curves for {v}: {e}")
        plt.close()

# --------------- PLOT TEST ACCURACY VS NHEAD COMPARISON ------------ #
try:
    fig = plt.figure(figsize=(6, 4))
    for v in variants:
        nheads, test_accs = [], []
        for nhead, res in (
            experiment_data.get(v, {}).get(dataset_name, {}).get("results", {}).items()
        ):
            nheads.append(int(nhead))
            test_accs.append(res["test_acc"])
        idx = np.argsort(nheads)
        plt.plot(np.array(nheads)[idx], np.array(test_accs)[idx], marker="o", label=v)
    plt.title(f"{dataset_name}: Test Accuracy vs nhead")
    plt.xlabel("Number of Attention Heads")
    plt.ylabel("Test Accuracy")
    plt.legend()
    save_path = os.path.join(working_dir, f"{dataset_name}_test_accuracy_vs_nhead.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error plotting test accuracy comparison: {e}")
    plt.close()

# ---------------------- PRINT SUMMARY METRICS ---------------------- #
for v in variants:
    res_dict = experiment_data.get(v, {}).get(dataset_name, {}).get("results", {})
    best = max(res_dict.items(), key=lambda x: x[1]["test_acc"]) if res_dict else None
    if best:
        print(f"{v:15s} | best test acc={best[1]['test_acc']:.4f} @ nhead={best[0]}")
