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
    exp = experiment_data["shuffle_only"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp:
    losses = exp["losses"]
    metrics = exp["metrics"]
    test = exp["test"]

    # --------------------------------------------------------------#
    # 1) Pre-training loss
    try:
        plt.figure()
        plt.plot(losses["pretrain"], marker="o", label="Pretrain CL Loss")
        plt.title("SPR_BENCH – Contrastive Pre-training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_pretrain_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating pretrain loss plot: {e}")
        plt.close()

    # --------------------------------------------------------------#
    # 2) Training vs Validation loss
    try:
        plt.figure()
        plt.plot(losses["train"], label="Train Loss")
        plt.plot(losses["val"], label="Validation Loss")
        plt.title("SPR_BENCH – Fine-tuning Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_train_val_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating train/val loss plot: {e}")
        plt.close()

    # --------------------------------------------------------------#
    # 3) Validation Accuracy & ACA
    try:
        plt.figure()
        plt.plot(metrics["val_acc"], label="Val Accuracy")
        plt.plot(metrics["val_aca"], label="Val ACA")
        plt.title("SPR_BENCH – Validation Accuracy (Left: Acc, Right: ACA)")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_acc_aca.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating val acc/aca plot: {e}")
        plt.close()

    # --------------------------------------------------------------#
    # 4) Final Test Metrics
    try:
        plt.figure()
        names = ["Acc", "SWA", "CWA", "ACA"]
        vals = [test.get("acc"), test.get("swa"), test.get("cwa"), test.get("aca")]
        plt.bar(names, vals)
        plt.ylim(0, 1)
        plt.title("SPR_BENCH – Test Metrics Overview")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

print("Plotting complete.")
