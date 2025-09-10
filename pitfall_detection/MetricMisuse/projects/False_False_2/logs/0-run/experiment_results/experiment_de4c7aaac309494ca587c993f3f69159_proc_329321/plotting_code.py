import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

hd_dict = experiment_data.get("hidden_dim_tuning", {})
hidden_dims = sorted([int(k) for k in hd_dict.keys()])


# helper to get list in same order
def get(hd, key_chain):
    d = hd_dict[str(hd)]
    for k in key_chain:
        d = d[k]
    return d


# 1) loss curves --------------------------------------------------------------
try:
    plt.figure(figsize=(6, 4))
    for hd in hidden_dims:
        epochs = get(hd, ["epochs"])
        train_loss = get(hd, ["losses", "train"])
        plt.plot(epochs, train_loss, label=f"hid {hd}")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss vs Epoch (Synthetic SPR)")
    plt.legend()
    fname = os.path.join(working_dir, "synthetic_spr_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) PHA curves ---------------------------------------------------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for hd in hidden_dims:
        epochs = get(hd, ["epochs"])
        tr_pha = get(hd, ["metrics", "train_PHA"])
        dv_pha = get(hd, ["metrics", "dev_PHA"])
        axes[0].plot(epochs, tr_pha, label=f"hid {hd}")
        axes[1].plot(epochs, dv_pha, label=f"hid {hd}")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[0].set_ylabel("PHA")
    axes[0].set_title("Left: Train PHA")
    axes[1].set_title("Right: Dev PHA")
    fig.suptitle("PHA Curves across Hidden Dimensions (Synthetic SPR)")
    for ax in axes:
        ax.legend()
    fname = os.path.join(working_dir, "synthetic_spr_pha_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating PHA curve plot: {e}")
    plt.close()

# 3) Test PHA vs hidden dim ---------------------------------------------------
best_hd, best_pha = None, -1.0
try:
    test_phas = []
    for hd in hidden_dims:
        pha = get(hd, ["test_metrics", "PHA"])
        test_phas.append(pha)
        if pha > best_pha:
            best_hd, best_pha = hd, pha
    plt.figure(figsize=(6, 4))
    plt.bar([str(hd) for hd in hidden_dims], test_phas, color="skyblue")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Test PHA")
    plt.title("Test PHA vs Hidden Dimension (Synthetic SPR)")
    fname = os.path.join(working_dir, "synthetic_spr_test_pha_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test PHA bar plot: {e}")
    plt.close()

# ------------------------------------------------------------
if best_hd is not None:
    tm = hd_dict[str(best_hd)]["test_metrics"]
    print(
        f"Best hidden dim = {best_hd} | SWA={tm['SWA']:.4f} "
        f"CWA={tm['CWA']:.4f} PHA={tm['PHA']:.4f}"
    )
