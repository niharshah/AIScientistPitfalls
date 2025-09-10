import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- Load experiment data ---------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["no_aug_pretraining"]["SPR_BENCH"]
    losses, metrics, test = exp["losses"], exp["metrics"], exp.get("test", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    losses, metrics, test = {}, {}, {}

# ---------------- Figure 1: pre-training loss ---------------- #
try:
    if losses.get("pretrain"):
        plt.figure()
        plt.plot(range(1, len(losses["pretrain"]) + 1), losses["pretrain"], marker="o")
        plt.title("SPR_BENCH: Contrastive Pre-training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        fname = os.path.join(working_dir, "SPR_BENCH_pretrain_loss.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating pretraining loss plot: {e}")
    plt.close()

# ---------------- Figure 2: train & val loss ---------------- #
try:
    if losses.get("train") and losses.get("val"):
        plt.figure()
        ep = range(1, len(losses["train"]) + 1)
        plt.plot(ep, losses["train"], label="Train")
        plt.plot(ep, losses["val"], label="Validation")
        plt.title("SPR_BENCH: Fine-tune Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_train_val_loss.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating train/val loss plot: {e}")
    plt.close()

# ---------------- Figure 3: val ACC & ACA ---------------- #
try:
    if metrics.get("val_acc") and metrics.get("val_aca"):
        plt.figure()
        ep = range(1, len(metrics["val_acc"]) + 1)
        plt.plot(ep, metrics["val_acc"], label="Val Acc")
        plt.plot(ep, metrics["val_aca"], label="Val ACA")
        plt.title("SPR_BENCH: Validation Accuracy & ACA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_acc_aca.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating val metrics plot: {e}")
    plt.close()

# ---------------- Figure 4: test metric bars ---------------- #
try:
    if test:
        labels = ["ACC", "SWA", "CWA", "ACA"]
        values = [test.get("acc"), test.get("swa"), test.get("cwa"), test.get("aca")]
        if all(v is not None for v in values):
            plt.figure()
            plt.bar(labels, values, color="skyblue")
            plt.ylim(0, 1)
            plt.title("SPR_BENCH: Test Metrics Summary")
            for idx, v in enumerate(values):
                plt.text(idx, v + 0.02, f"{v:.2f}", ha="center")
            fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test metrics plot: {e}")
    plt.close()

# ---------------- Print test metrics ---------------- #
if test:
    print("--- Test metrics ---")
    for k, v in test.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.4f}")
