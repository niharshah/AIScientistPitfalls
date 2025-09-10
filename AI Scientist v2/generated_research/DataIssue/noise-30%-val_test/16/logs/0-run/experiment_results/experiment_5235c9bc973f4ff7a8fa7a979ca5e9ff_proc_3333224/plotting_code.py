import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------------------------#
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load experiment results ---------------------------------#
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    ds_name = "SPR_BENCH"
    runs = experiment_data["batch_size"][ds_name]  # dict keyed by 'bs_XX'
    # Pre-extract arrays for easier plotting
    bs_values, epochs = [], None
    train_mcc, val_mcc, train_loss, val_loss, test_mcc = [], [], [], [], []
    for k, v in runs.items():
        bs = int(k.split("_")[1])
        bs_values.append(bs)
        if epochs is None:  # same for all
            epochs = v["epochs"]
        train_mcc.append(v["metrics"]["train_MCC"])
        val_mcc.append(v["metrics"]["val_MCC"])
        train_loss.append(v["losses"]["train"])
        val_loss.append(v["losses"]["val"])
        test_mcc.append(v["metrics"]["test_MCC"])

    # ------------- figure 1: MCC curves --------------------------------------#
    try:
        plt.figure()
        for bs, tr, vl in zip(bs_values, train_mcc, val_mcc):
            plt.plot(epochs, tr, "--", label=f"train bs={bs}")
            plt.plot(epochs, vl, "-", label=f"val   bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.title(f"{ds_name} MCC over Epochs\nLeft: Train (dashed), Right: Validation")
        plt.legend(fontsize=8, ncol=2)
        fname = os.path.join(working_dir, f"{ds_name}_MCC_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating MCC plot: {e}")
        plt.close()

    # ------------- figure 2: Loss curves -------------------------------------#
    try:
        plt.figure()
        for bs, tr, vl in zip(bs_values, train_loss, val_loss):
            plt.plot(epochs, tr, "--", label=f"train bs={bs}")
            plt.plot(epochs, vl, "-", label=f"val   bs={bs}")
        plt.xlabel("Epoch")
        plt.ylabel("Binary-Cross-Entropy Loss")
        plt.title(
            f"{ds_name} Loss over Epochs\nLeft: Train (dashed), Right: Validation"
        )
        plt.legend(fontsize=8, ncol=2)
        fname = os.path.join(working_dir, f"{ds_name}_Loss_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating Loss plot: {e}")
        plt.close()

    # ------------- figure 3: Test MCC bars -----------------------------------#
    try:
        plt.figure()
        plt.bar(range(len(bs_values)), test_mcc, tick_label=bs_values)
        plt.xlabel("Batch Size")
        plt.ylabel("Test MCC")
        plt.title(f"{ds_name} Final Test MCC per Batch Size")
        fname = os.path.join(working_dir, f"{ds_name}_Test_MCC_bar.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating Test MCC bar chart: {e}")
        plt.close()
