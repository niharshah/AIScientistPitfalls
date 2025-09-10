import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data.get("batch_size_tuning", {})
    bss = sorted(int(k.split("_")[1]) for k in runs.keys())
    run_keys = [f"bs_{bs}" for bs in bss]

    # Extract needed arrays
    train_losses = {rk: runs[rk]["losses"]["train"] for rk in run_keys}
    val_losses = {rk: runs[rk]["losses"]["val"] for rk in run_keys}
    val_hwa = {rk: [m["HWA"] for m in runs[rk]["metrics"]["val"]] for rk in run_keys}
    test_hwa = {rk: runs[rk]["test_metrics"]["HWA"] for rk in run_keys}

    # ---- Figure 1 : Loss curves ----
    try:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for rk in run_keys:
            axs[0].plot(train_losses[rk], label=rk)
            axs[1].plot(val_losses[rk], label=rk)
        axs[0].set_title("Train Loss")
        axs[1].set_title("Validation Loss")
        for ax in axs:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        fig.suptitle("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.tight_layout()
        path = os.path.join(working_dir, "SPR_BENCH_batch_size_loss_curves.png")
        plt.savefig(path)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating loss curve figure: {e}")
        plt.close()

    # ---- Figure 2 : Validation HWA ----
    try:
        plt.figure(figsize=(6, 4))
        for rk in run_keys:
            plt.plot(val_hwa[rk], label=rk)
        plt.title("SPR_BENCH Validation HWA across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(working_dir, "SPR_BENCH_validation_HWA_curves.png")
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating validation HWA figure: {e}")
        plt.close()

    # ---- Figure 3 : Test HWA bar chart ----
    try:
        plt.figure(figsize=(6, 4))
        bs_labels = [str(bs) for bs in bss]
        hwa_vals = [test_hwa[f"bs_{bs}"] for bs in bss]
        plt.bar(bs_labels, hwa_vals, color="skyblue")
        plt.title("SPR_BENCH Test HWA vs Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("HWA")
        for i, val in enumerate(hwa_vals):
            plt.text(i, val + 0.005, f"{val:.2f}", ha="center")
        plt.tight_layout()
        path = os.path.join(working_dir, "SPR_BENCH_test_HWA_bar.png")
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating test HWA bar figure: {e}")
        plt.close()

print("Done")
