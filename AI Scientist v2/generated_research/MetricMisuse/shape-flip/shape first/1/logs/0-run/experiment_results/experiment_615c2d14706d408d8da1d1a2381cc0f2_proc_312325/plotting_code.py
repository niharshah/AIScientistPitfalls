import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# load saved experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data.get("weight_decay", {})

    # --------------------------------------------------------
    # 1) Train / Val accuracy curves
    try:
        plt.figure()
        for run_key, run_val in runs.items():
            epochs = range(1, len(run_val["metrics"]["train_acc"]) + 1)
            plt.plot(epochs, run_val["metrics"]["train_acc"], label=f"{run_key} train")
            plt.plot(
                epochs,
                run_val["metrics"]["val_acc"],
                linestyle="--",
                label=f"{run_key} val",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR-BENCH: Train vs Val Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_train_val_accuracy.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # 2) Validation loss curves
    try:
        plt.figure()
        for run_key, run_val in runs.items():
            epochs = range(1, len(run_val["metrics"]["val_loss"]) + 1)
            plt.plot(epochs, run_val["metrics"]["val_loss"], label=run_key)
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.title("SPR-BENCH: Validation Loss Across Weight Decay Settings")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_val_loss.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating val-loss plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # 3) ZSRTA bar chart
    try:
        plt.figure()
        run_keys, zsrtas = [], []
        for run_key, run_val in runs.items():
            z_list = run_val["metrics"].get("ZSRTA", [])
            if len(z_list) > 0:
                run_keys.append(run_key)
                zsrtas.append(z_list[-1])
        if zsrtas:
            plt.bar(run_keys, zsrtas)
            plt.ylabel("ZSRTA")
            plt.title("SPR-BENCH: Zero-Shot Rule Transfer Accuracy by Weight Decay")
            plt.xticks(rotation=45)
            fname = os.path.join(working_dir, "spr_bench_zsrta_bar.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        else:
            print("No ZSRTA data found; skipping bar chart.")
    except Exception as e:
        print(f"Error creating ZSRTA plot: {e}")
        plt.close()
