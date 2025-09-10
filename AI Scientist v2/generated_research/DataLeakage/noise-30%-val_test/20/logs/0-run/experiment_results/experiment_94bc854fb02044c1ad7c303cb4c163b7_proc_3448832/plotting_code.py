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


# Helper: safely fetch nested dicts
def get(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


run_key, ds_key = "mean_pooling_no_cls", "SPR_BENCH"
loss_train = get(experiment_data, run_key, ds_key, "losses", "train", default=[])
loss_val = get(experiment_data, run_key, ds_key, "losses", "val", default=[])
metrics_val = get(experiment_data, run_key, ds_key, "metrics", "val", default=[])
preds = np.array(get(experiment_data, run_key, ds_key, "predictions", default=[]))
gts = np.array(get(experiment_data, run_key, ds_key, "ground_truth", default=[]))
epochs = np.arange(1, len(loss_val) + 1)

# 1) Loss curves
try:
    plt.figure()
    if loss_train:
        plt.plot(epochs, loss_train[: len(epochs)], label="Train")
    if loss_val:
        plt.plot(epochs, loss_val, label="Validation")
    plt.title("SPR_BENCH Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Macro-F1 curve
try:
    plt.figure()
    macro_f1 = [m.get("macro_f1") for m in metrics_val if m]
    if macro_f1:
        plt.plot(epochs, macro_f1, marker="o")
        plt.title("SPR_BENCH Macro-F1 over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_macro_f1_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Macro-F1 plot: {e}")
    plt.close()

# 3) CWA curve
try:
    plt.figure()
    cwa_vals = [m.get("cwa") for m in metrics_val if m]
    if cwa_vals:
        plt.plot(epochs, cwa_vals, marker="o", color="green")
        plt.title("SPR_BENCH Complexity-Weighted Accuracy (CWA)")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_cwa_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# 4) Ground truth vs prediction label distribution
try:
    if preds.size and gts.size:
        labels = sorted(set(np.concatenate([gts, preds])))
        gt_counts = [np.sum(gts == lbl) for lbl in labels]
        pr_counts = [np.sum(preds == lbl) for lbl in labels]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        axes[0].bar(labels, gt_counts, color="steelblue")
        axes[0].set_title("Ground Truth")
        axes[1].bar(labels, pr_counts, color="darkorange")
        axes[1].set_title("Predictions")
        for ax in axes:
            ax.set_xlabel("Label")
            ax.set_ylabel("Count")
        fig.suptitle(
            "SPR_BENCH Label Distribution\nLeft: Ground Truth, Right: Generated Samples",
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_label_distribution.png"))
        plt.close()
except Exception as e:
    print(f"Error creating label distribution plot: {e}")
    plt.close()
