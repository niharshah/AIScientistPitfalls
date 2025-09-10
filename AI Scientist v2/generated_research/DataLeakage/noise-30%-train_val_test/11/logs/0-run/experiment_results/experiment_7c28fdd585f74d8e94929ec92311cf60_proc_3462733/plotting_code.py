import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    lr_records = experiment_data["learning_rate"]["SPR_BENCH"]
    lrs_sorted = sorted(lr_records.keys(), key=lambda x: float(x))
    epochs = lr_records[lrs_sorted[0]]["epochs"]

    # --------- plot 1: loss curves -----------
    try:
        plt.figure()
        for lr in lrs_sorted:
            rec = lr_records[lr]
            plt.plot(epochs, rec["losses"]["train"], label=f"train lr={lr}")
            plt.plot(epochs, rec["losses"]["val"], linestyle="--", label=f"val lr={lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH – Loss Curves Across Learning Rates\nTrain (solid) vs Validation (dashed)"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves plot: {e}")
        plt.close()

    # --------- plot 2: macro-F1 curves -------
    try:
        plt.figure()
        for lr in lrs_sorted:
            rec = lr_records[lr]
            plt.plot(epochs, rec["metrics"]["train_macro_f1"], label=f"train lr={lr}")
            plt.plot(
                epochs,
                rec["metrics"]["val_macro_f1"],
                linestyle="--",
                label=f"val lr={lr}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title(
            "SPR_BENCH – Macro-F1 Curves Across Learning Rates\nTrain (solid) vs Validation (dashed)"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curves plot: {e}")
        plt.close()

    # --------- plot 3: test macro-F1 bar -----
    try:
        plt.figure()
        test_f1s = [lr_records[lr]["test_macro_f1"] for lr in lrs_sorted]
        plt.bar([str(lr) for lr in lrs_sorted], test_f1s)
        plt.xlabel("Learning Rate")
        plt.ylabel("Test Macro F1")
        plt.title("SPR_BENCH – Test Macro-F1 by Learning Rate")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_macroF1_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test F1 bar plot: {e}")
        plt.close()

    # --------- print evaluation summary ------
    best_lr = max(
        lrs_sorted, key=lambda lr: lr_records[lr]["metrics"]["val_macro_f1"][-1]
    )
    best_val = lr_records[best_lr]["metrics"]["val_macro_f1"][-1]
    best_test = lr_records[best_lr]["test_macro_f1"]
    print(
        f"Best LR (final val macro-F1): {best_lr} | Val F1: {best_val:.4f} | Test F1: {best_test:.4f}"
    )
