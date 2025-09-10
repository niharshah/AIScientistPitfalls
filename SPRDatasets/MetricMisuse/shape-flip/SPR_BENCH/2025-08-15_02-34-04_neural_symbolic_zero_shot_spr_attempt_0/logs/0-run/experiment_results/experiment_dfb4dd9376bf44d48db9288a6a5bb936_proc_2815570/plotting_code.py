import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None and "num_epochs" in exp:

    runs = exp["num_epochs"]

    # ------------- 1) combined loss curves -------------
    try:
        plt.figure(figsize=(6, 4))
        for run_name, run in runs.items():
            x = np.arange(len(run["losses"]["train"]))
            plt.plot(x, run["losses"]["train"], ls="--", label=f"{run_name}-train")
            plt.plot(x, run["losses"]["val"], ls="-", label=f"{run_name}-val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves\nTrain (dashed) vs Validation (solid)")
        plt.legend(fontsize=8)
        fname = os.path.join(working_dir, "spr_loss_curves_all_runs.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------- 2) HWA evolution -------------
    try:
        plt.figure(figsize=(6, 4))
        for run_name, run in runs.items():
            hwa_vals = [m[2] for m in run["metrics"]["val"]]
            # sample at most 50 points to keep figure readable
            step = max(1, len(hwa_vals) // 50)
            plt.plot(np.arange(len(hwa_vals))[::step], hwa_vals[::step], label=run_name)
        plt.xlabel("Epoch")
        plt.ylabel("Validation HWA")
        plt.title("SPR_BENCH Validation HWA Across Epochs")
        plt.legend(fontsize=8)
        fname = os.path.join(working_dir, "spr_val_hwa_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # ------------- 3) final test HWA bar chart -------------
    try:
        plt.figure(figsize=(6, 4))
        names, hwas = [], []
        for run_name, run in runs.items():
            names.append(run_name.replace("epochs_", "e"))
            hwas.append(run["metrics"]["test"][2])
        plt.bar(names, hwas, color="skyblue")
        plt.ylabel("Test HWA")
        plt.title("SPR_BENCH Final Test HWA by num_epochs Setting")
        fname = os.path.join(working_dir, "spr_test_hwa_bar.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()
