import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------- load data -------------------------
exp_path_opts = [
    os.path.join(working_dir, "experiment_data.npy"),
    os.path.join(os.getcwd(), "experiment_data.npy"),
]
for p in exp_path_opts:
    if os.path.exists(p):
        experiment_data = np.load(p, allow_pickle=True).item()
        break
else:
    raise FileNotFoundError("experiment_data.npy not found in expected locations.")

ed = experiment_data["num_epochs"]["SPR_BENCH"]
runs = len(ed["losses"]["train"])
epoch_cfgs = ed["epoch_config"]


# -------------- helper to limit epochs plotted ---------------
def maybe_sample_epochs(x):
    if len(x) > 50:  # unlikely, but stay safe
        step = len(x) // 50
        return x[::step]
    return x


# ------------------------- PLOT 1 -----------------------------
try:
    plt.figure()
    for i in range(runs):
        epochs = maybe_sample_epochs(ed["epochs"][i])
        plt.plot(epochs, ed["losses"]["train"][i], label=f"train_{epoch_cfgs[i]}ep")
        plt.plot(
            epochs,
            ed["losses"]["val"][i],
            linestyle="--",
            label=f"val_{epoch_cfgs[i]}ep",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs. Validation Loss")
    plt.legend()
    fpath = os.path.join(working_dir, "SPR_BENCH_train_val_loss.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------- PLOT 2 -----------------------------
try:
    plt.figure()
    for i in range(runs):
        epochs = maybe_sample_epochs(ed["epochs"][i])
        plt.plot(epochs, ed["metrics"]["train_f1"][i], label=f"train_{epoch_cfgs[i]}ep")
        plt.plot(
            epochs,
            ed["metrics"]["val_f1"][i],
            linestyle="--",
            label=f"val_{epoch_cfgs[i]}ep",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training vs. Validation Macro-F1")
    plt.legend()
    fpath = os.path.join(working_dir, "SPR_BENCH_train_val_f1.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# ------------------------- PLOT 3 -----------------------------
try:
    gt = np.array(ed["ground_truth"][0])  # use first run (all runs identical gt)
    preds = np.array(
        ed["predictions"][np.argmax([max(v) for v in ed["metrics"]["val_f1"]])]
    )
    labels = sorted(set(gt))
    gt_counts = [np.sum(gt == l) for l in labels]
    pr_counts = [np.sum(preds == l) for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].bar(labels, gt_counts, color="tab:blue")
    axes[0].set_title("Ground Truth")
    axes[1].bar(labels, pr_counts, color="tab:orange")
    axes[1].set_title("Model Predictions")
    fig.suptitle(
        "SPR_BENCH: Label Distribution (Left: Ground Truth, Right: Generated Samples)"
    )
    fpath = os.path.join(working_dir, "SPR_BENCH_label_distribution.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating label distribution plot: {e}")
    plt.close()

# ----------------------- METRIC REPORT ------------------------
print("\nBest Validation F1 per run:")
for i in range(runs):
    best_f1 = max(ed["metrics"]["val_f1"][i])
    print(f"  Run {i+1} (num_epochs={epoch_cfgs[i]}): best_val_F1={best_f1:.4f}")
