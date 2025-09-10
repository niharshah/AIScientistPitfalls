import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load experiment data ---
try:
    edict = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = edict["FT_EPOCHS"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

best_scwa = {}
if spr is not None:
    epochs_all = np.array(spr["epochs"])
    train_loss = np.array(spr["losses"]["train"])
    val_loss = np.array(spr["losses"]["val"])
    val_scwa = np.array(spr["metrics"]["val"])
    ft_setting = np.array(spr["ft_setting"])

    uniq_ft = sorted(set(ft_setting))
    # --- plot losses ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(
            "SPR_BENCH Training & Validation Loss\nLeft: Train Loss, Right: Validation Loss"
        )
        for ft in uniq_ft[:5]:  # plot at most 5 settings
            idx = ft_setting == ft
            axes[0].plot(epochs_all[idx], train_loss[idx], label=f"FT={ft}")
            axes[1].plot(epochs_all[idx], val_loss[idx], label=f"FT={ft}")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        fig.savefig(fname)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # --- plot SCWA metric ---
    try:
        plt.figure(figsize=(6, 4))
        plt.title("SPR_BENCH Validation SCWA vs Epochs")
        for ft in uniq_ft[:5]:
            idx = ft_setting == ft
            plt.plot(epochs_all[idx], val_scwa[idx], label=f"FT={ft}")
            best_scwa[ft] = val_scwa[idx].max()
        plt.xlabel("Epoch")
        plt.ylabel("SCWA")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_scwa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SCWA plot: {e}")
        plt.close()

# --- print best SCWA per fine-tuning setting ---
if best_scwa:
    print("Best validation SCWA per FT_EPOCHS setting:")
    for ft, score in sorted(best_scwa.items()):
        print(f"  FT={ft:2d}: {score:.4f}")
