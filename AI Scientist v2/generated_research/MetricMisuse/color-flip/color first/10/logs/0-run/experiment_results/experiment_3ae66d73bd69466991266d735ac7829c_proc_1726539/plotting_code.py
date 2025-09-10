import matplotlib.pyplot as plt
import numpy as np
import os

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

if experiment_data:
    data = experiment_data["learning_rate"]["SPR_BENCH"]
    lrs = data["lrs"]
    train_losses_runs = data["losses"]["train"]
    val_losses_runs = data["losses"]["val"]
    val_cwa_runs = data["metrics"]["val_CompWA"]

    # ---------- Plot 1: Loss curves ----------
    try:
        plt.figure()
        for lr, tr_ls, va_ls in zip(lrs, train_losses_runs, val_losses_runs):
            epochs = np.arange(1, len(tr_ls) + 1)
            plt.plot(epochs, tr_ls, "--", label=f"Train (lr={lr})")
            plt.plot(epochs, va_ls, "-", label=f"Val (lr={lr})")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train (--), Right: Val (-)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- Plot 2: CompWA curves ----------
    try:
        plt.figure()
        for lr, cwa_ls in zip(lrs, val_cwa_runs):
            epochs = np.arange(1, len(cwa_ls) + 1)
            plt.plot(epochs, cwa_ls, label=f"Val CompWA (lr={lr})")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title("SPR_BENCH Validation CompWA Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_compwa_curves.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating CompWA curve plot: {e}")
        plt.close()

    # ---------- Plot 3: Final metrics per LR ----------
    try:
        final_val_losses = [ls[-1] for ls in val_losses_runs]
        final_val_cwas = [cw[-1] for cw in val_cwa_runs]
        x = np.arange(len(lrs))

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.bar(
            x - 0.2,
            final_val_losses,
            width=0.4,
            color="tab:blue",
            label="Final Val Loss",
        )
        ax2.bar(
            x + 0.2,
            final_val_cwas,
            width=0.4,
            color="tab:orange",
            label="Final Val CompWA",
        )

        ax1.set_xlabel("Learning Rate Index")
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(lr) for lr in lrs])
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("CompWA")
        plt.title(
            "SPR_BENCH Final Metrics vs Learning Rate\nLeft: Loss (blue), Right: CompWA (orange)"
        )
        fig.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_lr_performance.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating LR performance plot: {e}")
        plt.close()
