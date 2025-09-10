import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    runs = experiment_data["learning_rate"]["SPR_BENCH"]["runs"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = []

if runs:  # proceed only if data is present
    # Gather common arrays
    lrs = [r["lr"] for r in runs]
    epochs = runs[0]["epochs"] if runs else []
    train_losses = [r["losses"]["train"] for r in runs]
    val_losses = [r["losses"]["val"] for r in runs]
    train_f1s = [r["metrics"]["train"] for r in runs]
    val_f1s = [r["metrics"]["val"] for r in runs]
    test_f1s = [r["test_macroF1"] for r in runs]

    # ---------- Plot 1: Loss curves ----------
    try:
        plt.figure()
        for i, lr in enumerate(lrs):
            plt.plot(epochs, train_losses[i], label=f"train_lr={lr}")
            plt.plot(epochs, val_losses[i], "--", label=f"val_lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Val Loss for different Learning Rates")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ---------- Plot 2: Macro-F1 curves ----------
    try:
        plt.figure()
        for i, lr in enumerate(lrs):
            plt.plot(epochs, train_f1s[i], label=f"train_lr={lr}")
            plt.plot(epochs, val_f1s[i], "--", label=f"val_lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Train vs Val Macro-F1 for different Learning Rates")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curves: {e}")
        plt.close()

    # ---------- Plot 3: Test Macro-F1 bar chart ----------
    try:
        plt.figure()
        plt.bar(range(len(lrs)), test_f1s, tick_label=[f"{lr:.0e}" for lr in lrs])
        plt.ylabel("Test Macro-F1")
        plt.title("SPR_BENCH: Test Macro-F1 vs Learning Rate")
        fname = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test F1 bar: {e}")
        plt.close()

    # ---------- Print best LR ----------
    best_idx = int(np.argmax(test_f1s))
    print(f"Best LR={lrs[best_idx]} with Test Macro-F1={test_f1s[best_idx]:.4f}")
