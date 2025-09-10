import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data -------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    spr_data = experiment_data["learning_rate"]["SPR_BENCH"]
    lrs = np.array(spr_data["lr_values"])
    val_f1_runs = spr_data["metrics"]["val_f1"]  # list of lists
    final_val_f1 = np.array([vals[-1] for vals in val_f1_runs])

    # identify best lr
    best_idx = int(np.argmax(final_val_f1))
    best_lr = lrs[best_idx]
    best_epochs = spr_data["epochs_record"][best_idx]
    best_train_loss = spr_data["losses"]["train"][best_idx]
    best_val_loss = spr_data["losses"]["val"][best_idx]
    best_train_f1 = spr_data["metrics"]["train_f1"][best_idx]
    best_val_f1 = spr_data["metrics"]["val_f1"][best_idx]

    # ------------- plot 1: LR sweep summary -------------
    try:
        plt.figure()
        plt.plot(lrs, final_val_f1, marker="o")
        plt.xscale("log")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Final Dev Macro-F1")
        plt.title("SPR_BENCH: Final Dev Macro-F1 vs Learning Rate")
        fname = os.path.join(working_dir, "SPR_BENCH_lr_sweep_final_f1.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating LR sweep plot: {e}")
        plt.close()

    # ------------- plot 2: best LR loss curves ----------
    try:
        plt.figure()
        plt.plot(best_epochs, best_train_loss, label="Train Loss")
        plt.plot(best_epochs, best_val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Loss Curves (Best LR={best_lr})")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_bestLR_{best_lr}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ------------- plot 3: best LR F1 curves ------------
    try:
        plt.figure()
        plt.plot(best_epochs, best_train_f1, label="Train Macro-F1")
        plt.plot(best_epochs, best_val_f1, label="Val Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"SPR_BENCH Macro-F1 Curves (Best LR={best_lr})")
        plt.legend()
        fname = os.path.join(working_dir, f"SPR_BENCH_bestLR_{best_lr}_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curves: {e}")
        plt.close()

    print(f"Best LR {best_lr} with final dev Macro-F1 {final_val_f1[best_idx]:.4f}")
