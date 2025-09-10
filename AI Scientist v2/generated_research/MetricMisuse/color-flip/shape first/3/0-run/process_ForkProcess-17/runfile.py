import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

run = experiment_data.get("MLM_pretrain", {}).get("SPR_BENCH", {})

losses = run.get("losses", {})
metrics = run.get("metrics", {})
test_res = run.get("test", {})

# ----------------- plot 1: loss curves ------------------
try:
    plt.figure()
    if "pretrain" in losses:
        plt.plot(
            range(1, len(losses["pretrain"]) + 1),
            losses["pretrain"],
            label="Pre-train MLM",
        )
    if "train" in losses:
        # offset fine-tuning epochs so the x-axis is continuous
        offset = len(losses.get("pretrain", []))
        xs = list(range(1 + offset, len(losses["train"]) + 1 + offset))
        plt.plot(xs, losses["train"], label="Fine-tune Train")
    if "val" in losses:
        offset = len(losses.get("pretrain", []))
        xs = list(range(1 + offset, len(losses["val"]) + 1 + offset))
        plt.plot(xs, losses["val"], label="Fine-tune Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Loss Curves (MLM & Fine-tune)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ----------------- plot 2: validation metrics -----------
try:
    plt.figure()
    if "val_acc" in metrics:
        plt.plot(
            range(1, len(metrics["val_acc"]) + 1), metrics["val_acc"], label="Val ACC"
        )
    if "val_aca" in metrics:
        plt.plot(
            range(1, len(metrics["val_aca"]) + 1), metrics["val_aca"], label="Val ACA"
        )
    plt.xlabel("Fine-tune Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Validation Accuracy Metrics")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating val metric plot: {e}")
    plt.close()

# ----------------- plot 3: test metrics -----------------
try:
    plt.figure()
    keys = ["acc", "swa", "cwa", "aca"]
    vals = [test_res[k] for k in keys if k in test_res]
    shown_keys = [k.upper() for k in keys if k in test_res]
    plt.bar(shown_keys, vals)
    plt.ylim(0, 1)
    plt.title("SPR_BENCH: Test Metrics")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric plot: {e}")
    plt.close()

# ----------------- print exact test numbers -------------
if test_res:
    for k, v in test_res.items():
        if isinstance(v, (int, float)):
            print(f"{k.upper()}: {v:.4f}")
