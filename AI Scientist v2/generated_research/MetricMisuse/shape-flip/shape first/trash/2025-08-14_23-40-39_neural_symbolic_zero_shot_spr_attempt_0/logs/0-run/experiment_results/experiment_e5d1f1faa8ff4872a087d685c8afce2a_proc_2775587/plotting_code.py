import matplotlib.pyplot as plt
import numpy as np
import os

# ----- setup -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    ds_data = experiment_data["BATCH_SIZE"]["SPR_BENCH"]
    batch_sizes = ds_data["batch_sizes"]
    per_bs = ds_data["per_bs_metrics"]
    final_val_loss = ds_data["metrics"]["val_loss"]
    final_val_bps = ds_data["metrics"]["val_bps"]

    # ---------- per-batch-size training curves (max 4 figs) ----------
    for i, bs in enumerate(batch_sizes):
        try:
            metrics = per_bs[i]
            epochs = np.arange(1, len(metrics["train_loss"]) + 1)
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, metrics["train_loss"], marker="o", label="Train Loss")
            plt.plot(epochs, metrics["val_loss"], marker="o", label="Val Loss")
            plt.plot(epochs, metrics["bps"], marker="o", label="Val BPS")
            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.title(f"SPR_BENCH Training Curves (bs={bs})")
            plt.legend()
            fname = f"SPR_BENCH_training_curves_bs{bs}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting curves for bs={bs}: {e}")
            plt.close()

    # ---------- aggregate figure (1 fig) ----------
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(batch_sizes, final_val_bps, marker="o", label="Final Val BPS")
        plt.plot(batch_sizes, final_val_loss, marker="o", label="Final Val Loss")
        plt.xlabel("Batch Size")
        plt.ylabel("Metric Value")
        plt.title("SPR_BENCH Final Validation Metrics vs Batch Size")
        plt.legend()
        fname = "SPR_BENCH_val_metrics_vs_batch_size.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error plotting aggregate metrics: {e}")
        plt.close()

    # ---------- quick console summary ----------
    print("Final validation results:")
    for bs, loss, bps in zip(batch_sizes, final_val_loss, final_val_bps):
        print(f"  bs={bs}: val_loss={loss:.4f}, val_bps={bps:.3f}")
