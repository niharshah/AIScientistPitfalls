import matplotlib.pyplot as plt
import numpy as np
import os

# setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data -------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    exp = experiment_data["nhead_tuning"]["SPR_BENCH"]
    nheads = exp["nhead_values"]
    epochs_list = exp["epochs"]  # list of epoch indices for each cfg
    train_f1 = exp["metrics"]["train_macro_f1"]
    val_f1 = exp["metrics"]["val_macro_f1"]
    train_l = exp["losses"]["train"]
    val_l = exp["losses"]["val"]
    test_f1 = exp["metrics"]["test_macro_f1"]

    # 1. F1 curves -------------------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        for i, nh in enumerate(nheads):
            plt.plot(epochs_list[i], train_f1[i], "--", label=f"train nhead={nh}")
            plt.plot(epochs_list[i], val_f1[i], "-", label=f"val nhead={nh}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("SPR_BENCH Train vs Val Macro F1 (nhead tuning)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_macro_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # 2. Loss curves -----------------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        for i, nh in enumerate(nheads):
            plt.plot(epochs_list[i], train_l[i], "--", label=f"train nhead={nh}")
            plt.plot(epochs_list[i], val_l[i], "-", label=f"val nhead={nh}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Train vs Val Loss (nhead tuning)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # 3. Test macro-F1 bar chart ----------------------------------------------
    try:
        plt.figure(figsize=(5, 3))
        plt.bar([str(nh) for nh in nheads], test_f1, color="skyblue")
        plt.xlabel("nhead")
        plt.ylabel("Test Macro F1")
        plt.title("SPR_BENCH Test Macro F1 by nhead")
        for i, v in enumerate(test_f1):
            plt.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_macro_f1_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test F1 bar plot: {e}")
        plt.close()

    # 4. Best-config detailed curves ------------------------------------------
    try:
        # pick config with best peak val-F1
        best_idx = int(np.argmax([max(v) for v in val_f1]))
        e = epochs_list[best_idx]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        # Left: loss
        axes[0].plot(e, train_l[best_idx], "--", label="train")
        axes[0].plot(e, val_l[best_idx], "-", label="val")
        axes[0].set_title("Left: Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend(fontsize=7)
        # Right: F1
        axes[1].plot(e, train_f1[best_idx], "--", label="train")
        axes[1].plot(e, val_f1[best_idx], "-", label="val")
        axes[1].set_title("Right: Macro F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro F1")
        axes[1].legend(fontsize=7)
        fig.suptitle(
            f"SPR_BENCH Best Config nhead={nheads[best_idx]} "
            "(Left: Loss, Right: F1)",
            fontsize=10,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = os.path.join(working_dir, "SPR_BENCH_best_config_learning_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating best-config plot: {e}")
        plt.close()

    # print simple evaluation --------------------------------------------------
    best_nhead = nheads[int(np.argmax(test_f1))]
    print(f"Best nhead on TEST macro-F1: {best_nhead} => {max(test_f1):.4f}")
