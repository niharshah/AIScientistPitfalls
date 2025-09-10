import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    ed = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = ed["learning_rate_sweep"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

if ed is not None:
    lrs = ed["lr_vals"]
    epochs_all = ed["epochs"]
    tr_losses, va_losses = ed["losses"]["train"], ed["losses"]["val"]
    tr_f1s, va_f1s = ed["metrics"]["train_f1"], ed["metrics"]["val_f1"]

    # 1) Loss curves -----------------------------------------------------------
    try:
        plt.figure()
        for lr, ep, tl, vl in zip(lrs, epochs_all, tr_losses, va_losses):
            plt.plot(ep, tl, label=f"train lr={lr:.0e}")
            plt.plot(ep, vl, "--", label=f"val lr={lr:.0e}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train/Val Loss vs Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # 2) F1 curves -------------------------------------------------------------
    try:
        plt.figure()
        for lr, ep, tf, vf in zip(lrs, epochs_all, tr_f1s, va_f1s):
            plt.plot(ep, tf, label=f"train lr={lr:.0e}")
            plt.plot(ep, vf, "--", label=f"val lr={lr:.0e}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Train/Val F1 vs Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curves: {e}")
        plt.close()

    # 3) Bar chart of final val F1 --------------------------------------------
    final_val_f1 = [vf[-1] for vf in va_f1s]
    try:
        plt.figure()
        plt.bar([f"{lr:.0e}" for lr in lrs], final_val_f1, color="skyblue")
        plt.xlabel("Learning Rate")
        plt.ylabel("Final Val Macro-F1")
        plt.title("SPR_BENCH: Final Validation F1 by Learning Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_val_f1_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()

    # 4) Confusion matrix on test set -----------------------------------------
    preds, gts = np.array(ed["predictions"]), np.array(ed["ground_truth"])
    labels_sorted = np.unique(gts)
    try:
        cm = confusion_matrix(gts, preds, labels=labels_sorted)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(labels_sorted)), labels_sorted)
        plt.yticks(range(len(labels_sorted)), labels_sorted)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix (Best LR)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------- Print summary metrics -----------------------------------------
    print("Final Validation F1 per LR:")
    for lr, f1 in zip(lrs, final_val_f1):
        print(f"  lr={lr:.0e}: val_F1={f1:.4f}")
    best_idx = int(np.argmax(final_val_f1))
    print(f"Best LR = {lrs[best_idx]:.0e}")
    test_macro_f1 = (
        f1_score(gts, preds, average="macro") if len(preds) else float("nan")
    )
    print(f"Test Macro-F1 with best LR: {test_macro_f1:.4f}")
