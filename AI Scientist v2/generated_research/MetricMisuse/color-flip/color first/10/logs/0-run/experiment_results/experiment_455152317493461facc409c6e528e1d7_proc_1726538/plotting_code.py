import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    exp = experiment_data["num_epochs"]["SPR_BENCH"]
    cfg_vals = exp["config_values"]
    train_losses_all = exp["losses"]["train"]
    val_losses_all = exp["losses"]["val"]
    val_cwa_all = exp["metrics"]["val_CompWA"]

    # helper to pad lists with np.nan so they align in length
    def pad(seq, L):
        return np.array(seq + [np.nan] * (L - len(seq)))

    max_len = max(len(x) for x in val_losses_all)

    # -------- Plot 1: Loss curves ----------
    try:
        plt.figure()
        for cfg, tr, va in zip(cfg_vals, train_losses_all, val_losses_all):
            x = np.arange(1, len(tr) + 1)
            plt.plot(x, tr, "--", label=f"train (E{cfg})")
            plt.plot(x, va, "-", label=f"val (E{cfg})")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend(fontsize=7)
        save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # -------- Plot 2: Validation CompWA ----------
    try:
        plt.figure()
        for cfg, cwa in zip(cfg_vals, val_cwa_all):
            x = np.arange(1, len(cwa) + 1)
            plt.plot(x, cwa, label=f"val_CompWA (E{cfg})")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title("SPR_BENCH: Validation CompWA across Epochs")
        plt.legend(fontsize=7)
        save_path = os.path.join(working_dir, "SPR_BENCH_val_compwa_curves.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA curves: {e}")
        plt.close()

    # -------- Plot 3: Best metrics vs hyper-parameter ----------
    try:
        best_val_losses = [min(v) if len(v) else np.nan for v in val_losses_all]
        best_val_cwas = [max(c) if len(c) else np.nan for c in val_cwa_all]

        x = np.arange(len(cfg_vals))
        width = 0.35

        plt.figure()
        plt.bar(x - width / 2, best_val_losses, width, label="Best Val Loss")
        plt.bar(x + width / 2, best_val_cwas, width, label="Best Val CompWA")
        plt.xticks(ticks=x, labels=cfg_vals)
        plt.xlabel("max_epochs")
        plt.title("SPR_BENCH: Best Metrics per max_epochs Setting")
        plt.legend()
        save_path = os.path.join(working_dir, "SPR_BENCH_best_metrics_bar.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Error creating best metric bar plot: {e}")
        plt.close()

    # -------- Console summary ----------
    print("\n=== Best validation metrics per max_epochs ===")
    for cfg, bl, bcwa in zip(cfg_vals, best_val_losses, best_val_cwas):
        print(
            f"max_epochs={cfg:<3} | best_val_loss={bl:0.4f} | best_val_CompWA={bcwa:0.4f}"
        )
