import matplotlib.pyplot as plt
import numpy as np
import os

# recreate working directory path
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    dstore = experiment_data["weight_decay_tuning"]["SPR_BENCH"]
    wds = dstore["wd_values"]
    epochs_per_run = dstore["epochs"]  # list[list]
    train_losses = dstore["losses"]["train"]  # list[list]
    val_losses = dstore["losses"]["val"]
    train_f1s = dstore["metrics"]["train"]
    val_f1s = dstore["metrics"]["val"]
    test_macro_f1 = dstore["test_macroF1"]

    # ----------------- Loss curves ---------------------------------
    try:
        plt.figure()
        for wd, ep, tl, vl in zip(wds, epochs_per_run, train_losses, val_losses):
            plt.plot(ep, tl, label=f"Train wd={wd}")
            plt.plot(ep, vl, linestyle="--", label=f"Val wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves (Train vs Val)")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ----------------- F1 curves -----------------------------------
    try:
        plt.figure()
        for wd, ep, tf, vf in zip(wds, epochs_per_run, train_f1s, val_f1s):
            plt.plot(ep, tf, label=f"Train wd={wd}")
            plt.plot(ep, vf, linestyle="--", label=f"Val wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 Curves (Train vs Val)")
        plt.legend(fontsize=7)
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve plot: {e}")
        plt.close()

    # ----------------- Test Macro-F1 bar chart ---------------------
    try:
        plt.figure()
        plt.bar([str(wd) for wd in wds], test_macro_f1, color="skyblue")
        plt.xlabel("Weight Decay")
        plt.ylabel("Test Macro-F1")
        plt.title("SPR_BENCH Test Macro-F1 vs Weight Decay")
        for idx, val in enumerate(test_macro_f1):
            plt.text(idx, val + 0.01, f"{val:.2f}", ha="center", fontsize=7)
        fname = os.path.join(working_dir, "SPR_BENCH_test_macroF1.png")
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error creating test Macro-F1 bar plot: {e}")
        plt.close()
