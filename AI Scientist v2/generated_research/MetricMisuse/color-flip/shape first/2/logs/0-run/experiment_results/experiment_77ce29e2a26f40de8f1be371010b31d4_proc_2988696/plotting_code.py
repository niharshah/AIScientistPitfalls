import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    losses_tr = experiment_data["dropout_rate"]["SPR_BENCH"]["losses"]["train"]
    losses_va = experiment_data["dropout_rate"]["SPR_BENCH"]["losses"]["val"]
    metrics_va = experiment_data["dropout_rate"]["SPR_BENCH"]["metrics"]["val"]

    # Helper to pivot lists -> dict[dropout] -> list ordered by epoch
    def pivot(records, idx_val=2):
        out = {}
        for dr, ep, val in records:
            out.setdefault(dr, {})[ep] = val
        return {k: [v[e] for e in sorted(v)] for k, v in out.items()}

    train_loss = pivot(losses_tr)
    val_loss = pivot(losses_va)
    hwa_dict = {}
    for m in metrics_va:
        hwa_dict.setdefault(m["dropout"], {})[m["epoch"]] = m["hwa"]
    hwa = {k: [v[e] for e in sorted(v)] for k, v in hwa_dict.items()}

    epochs = range(1, len(next(iter(train_loss.values()))) + 1)

    # -------- FIGURE 1: Loss curves --------
    try:
        plt.figure()
        for dr in sorted(train_loss):
            plt.plot(epochs, train_loss[dr], "--", label=f"train d={dr}")
            plt.plot(epochs, val_loss[dr], "-", label=f"val d={dr}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH BiLSTM Loss Curves\nLeft: Train (dashed), Right: Val (solid)"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------- FIGURE 2: HWA curves --------
    try:
        plt.figure()
        for dr in sorted(hwa):
            plt.plot(epochs, hwa[dr], marker="o", label=f"dropout={dr}")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Acc (HWA)")
        plt.title("SPR_BENCH Validation HWA Across Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # -------- FIGURE 3: Final-epoch HWA vs Dropout --------
    try:
        plt.figure()
        final_hwa = {dr: vals[-1] for dr, vals in hwa.items()}
        drs = list(sorted(final_hwa))
        vals = [final_hwa[d] for d in drs]
        plt.bar([str(d) for d in drs], vals)
        plt.xlabel("Dropout Rate")
        plt.ylabel("Final Epoch HWA")
        plt.title("SPR_BENCH Final-Epoch HWA by Dropout Setting")
        fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final HWA bar plot: {e}")
        plt.close()

    # -------- print best dropout --------
    best_dr = max(final_hwa, key=final_hwa.get)
    print(
        f"Best dropout (by final-epoch HWA): {best_dr}  -> HWA={final_hwa[best_dr]:.4f}"
    )
