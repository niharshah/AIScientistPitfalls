import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ed = experiment_data["learning_rate"]["SPR_BENCH"]
    lrs = ed["lrs"]
    epochs_lists = ed["epochs"]
    tr_losses = ed["losses"]["train"]
    val_losses = ed["losses"]["val"]
    tr_f1s = ed["metrics"]["train_f1"]
    val_f1s = ed["metrics"]["val_f1"]
    best_val_f1 = ed["best_val_f1"]
    preds_by_lr = ed["predictions"]
    gts = ed["ground_truth"]
    test_f1 = [f1_score(gts, preds, average="macro") for preds in preds_by_lr]

    # print summary table
    print("LR\tBestValF1\tTestF1")
    for lr, bv, tf in zip(lrs, best_val_f1, test_f1):
        print(f"{lr:.0e}\t{bv:.4f}\t{tf:.4f}")

    # ---------- Plot 1: Loss curves ----------
    try:
        plt.figure()
        for lr, ep, tl, vl in zip(lrs, epochs_lists, tr_losses, val_losses):
            plt.plot(ep, tl, label=f"Train lr={lr:.0e}")
            plt.plot(ep, vl, "--", label=f"Val lr={lr:.0e}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend(fontsize=8)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- Plot 2: F1 curves ----------
    try:
        plt.figure()
        for lr, ep, tf1, vf1 in zip(lrs, epochs_lists, tr_f1s, val_f1s):
            plt.plot(ep, tf1, label=f"Train lr={lr:.0e}")
            plt.plot(ep, vf1, "--", label=f"Val lr={lr:.0e}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("SPR_BENCH: Training vs Validation F1")
        plt.legend(fontsize=8)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    # ---------- Plot 3: Best val F1 vs LR ----------
    try:
        plt.figure()
        x = range(len(lrs))
        plt.bar(x, best_val_f1, tick_label=[f"{lr:.0e}" for lr in lrs])
        plt.ylabel("Best Validation Macro F1")
        plt.xlabel("Learning Rate")
        plt.title("SPR_BENCH: Best Validation F1 by Learning Rate")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_valF1_vs_lr.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating val-F1 vs LR plot: {e}")
        plt.close()

    # ---------- Plot 4: Test F1 vs LR ----------
    try:
        plt.figure()
        x = range(len(lrs))
        plt.bar(x, test_f1, tick_label=[f"{lr:.0e}" for lr in lrs])
        plt.ylabel("Test Macro F1")
        plt.xlabel("Learning Rate")
        plt.title("SPR_BENCH: Test F1 by Learning Rate")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_testF1_vs_lr.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test-F1 vs LR plot: {e}")
        plt.close()
