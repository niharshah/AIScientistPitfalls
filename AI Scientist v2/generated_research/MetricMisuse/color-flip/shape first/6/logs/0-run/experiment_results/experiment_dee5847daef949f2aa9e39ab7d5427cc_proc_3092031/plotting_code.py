import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------- load experiment data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["hidden_dim_tuning"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = {}

hidden_dims = sorted(int(k) for k in spr_data.keys())

colors = plt.cm.tab10.colors


# ------------ helper to fetch arrays ------------
def get_arr(hd, key1, key2):
    return np.array(spr_data[str(hd)][key1][key2])


# ------------ PLOT 1 : loss curves --------------
try:
    plt.figure()
    for i, hd in enumerate(hidden_dims):
        tr_loss = get_arr(hd, "losses", "train")
        val_loss = get_arr(hd, "losses", "val")
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.plot(
            epochs, tr_loss, color=colors[i % 10], linestyle="--", label=f"{hd}-train"
        )
        plt.plot(
            epochs, val_loss, color=colors[i % 10], linestyle="-", label=f"{hd}-val"
        )
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Loss Curves\nLeft: Train (--), Right: Validation (—)")
    plt.legend(fontsize=8, ncol=2)
    fname = os.path.join(working_dir, "SPR_loss_vs_epoch_hidden_dims.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------ PLOT 2 : CWA curves ---------------
try:
    plt.figure()
    for i, hd in enumerate(hidden_dims):
        val_cwa = get_arr(hd, "metrics", "val")
        epochs = np.arange(1, len(val_cwa) + 1)
        plt.plot(epochs, val_cwa, color=colors[i % 10], label=f"hd={hd}")
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("Color-Weighted Accuracy")
    plt.title("SPR Validation CWA Across Epochs\nDataset: SPR")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_val_CWA_vs_epoch_hidden_dims.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# ------------ PLOT 3 : AIS curves ---------------
try:
    plt.figure()
    for i, hd in enumerate(hidden_dims):
        ais = get_arr(hd, "AIS", "val")
        epochs = np.arange(1, len(ais) + 1)
        plt.plot(epochs, ais, color=colors[i % 10], label=f"hd={hd}")
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("AIS")
    plt.title("SPR Validation AIS Across Epochs\nDataset: SPR")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_val_AIS_vs_epoch_hidden_dims.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating AIS plot: {e}")
    plt.close()

# ------------ Print best CWA per hidden dim -------
for hd in hidden_dims:
    best_cwa = get_arr(hd, "metrics", "val").max()
    print(f"Hidden dim {hd:>4}: best Val CWA = {best_cwa:.3f}")
