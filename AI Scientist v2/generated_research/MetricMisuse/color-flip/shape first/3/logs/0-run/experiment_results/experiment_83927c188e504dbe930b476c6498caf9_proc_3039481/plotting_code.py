import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------#
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper for safe dict access
def safe_get(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


ds_key = "SPR_BENCH"
exp_keys = ["full_tune", "frozen_encoder"]

# 1. Contrastive pre-training loss (only in full_tune)
try:
    losses = safe_get(experiment_data, "full_tune", ds_key, "losses", "pretrain")
    if losses:
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses, marker="o")
        plt.title("Contrastive Pre-training Loss (SPR_BENCH)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        fname = os.path.join(working_dir, "spr_bench_pretrain_loss.png")
        plt.savefig(fname)
        print("Saved:", fname)
        plt.close()
except Exception as e:
    print(f"Error creating pretrain plot: {e}")
    plt.close()

# 2. Train vs Val loss (full_tune)
try:
    tr = safe_get(experiment_data, "full_tune", ds_key, "losses", "train")
    vl = safe_get(experiment_data, "full_tune", ds_key, "losses", "val")
    if tr and vl:
        plt.figure()
        plt.plot(range(1, len(tr) + 1), tr, label="Train")
        plt.plot(range(1, len(vl) + 1), vl, label="Val")
        plt.legend()
        plt.title("Full-tune Loss Curves (SPR_BENCH)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        fname = os.path.join(working_dir, "spr_bench_full_tune_loss_curves.png")
        plt.savefig(fname)
        print("Saved:", fname)
        plt.close()
except Exception as e:
    print(f"Error creating full_tune loss plot: {e}")
    plt.close()

# 3. Validation accuracy (full_tune)
try:
    acc = safe_get(experiment_data, "full_tune", ds_key, "metrics", "val_acc")
    if acc:
        plt.figure()
        plt.plot(range(1, len(acc) + 1), acc, marker="s", color="green")
        plt.title("Validation Accuracy (Full-tune, SPR_BENCH)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        fname = os.path.join(working_dir, "spr_bench_full_tune_val_acc.png")
        plt.savefig(fname)
        print("Saved:", fname)
        plt.close()
except Exception as e:
    print(f"Error creating val_acc plot: {e}")
    plt.close()

# 4. Validation ACA for both settings
try:
    plt.figure()
    for k in exp_keys:
        aca = safe_get(experiment_data, k, ds_key, "metrics", "val_aca")
        if aca:
            plt.plot(range(1, len(aca) + 1), aca, label=k)
    if plt.gca().has_data():
        plt.legend()
        plt.title("Validation ACA (SPR_BENCH)")
        plt.xlabel("Epoch")
        plt.ylabel("ACA")
        fname = os.path.join(working_dir, "spr_bench_val_aca_comparison.png")
        plt.savefig(fname)
        print("Saved:", fname)
    plt.close()
except Exception as e:
    print(f"Error creating ACA plot: {e}")
    plt.close()

# 5. Final test metrics bar chart
try:
    metrics = ["acc", "swa", "cwa", "aca"]
    width = 0.35
    x = np.arange(len(metrics))
    fig, ax = plt.subplots()
    plotted = False
    for i, k in enumerate(exp_keys):
        vals = [
            safe_get(experiment_data, k, ds_key, "test", m, default=np.nan)
            for m in metrics
        ]
        if not all(np.isnan(vals)):
            ax.bar(x + i * width, vals, width, label=k)
            plotted = True
    if plotted:
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title("Test Metrics Comparison (SPR_BENCH)")
        ax.legend()
        fname = os.path.join(working_dir, "spr_bench_test_metrics_comparison.png")
        plt.savefig(fname)
        print("Saved:", fname)
    plt.close(fig)
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
    plt.close()
