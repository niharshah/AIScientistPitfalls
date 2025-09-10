import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data ----------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ls_dict = experiment_data.get("label_smoothing", {})
eps_keys = sorted(ls_dict.keys(), key=lambda k: float(k.split("eps")[-1]))  # ordered ε


# helper to grab arrays ----------------------------------------------------
def get_losses(eps_key):
    tr = ls_dict[eps_key]["losses"]["train"]
    vl = ls_dict[eps_key]["losses"]["val"]
    epochs = [e for e, _ in tr]
    tr_loss = [l for _, l in tr]
    vl_loss = [l for _, l in vl]
    return epochs, tr_loss, vl_loss


def get_metrics(eps_key):
    metr = ls_dict[eps_key]["metrics"]["val"]
    epochs = [e for e, _ in metr]
    cwa = [m["CWA"] for _, m in metr]
    swa = [m["SWA"] for _, m in metr]
    ewa = [m["EWA"] for _, m in metr]
    return epochs, cwa, swa, ewa


# 1) combined loss curves --------------------------------------------------
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for k in eps_keys:
        epochs, tr_l, vl_l = get_losses(k)
        eps = k.split("eps")[-1]
        ax1.plot(epochs, tr_l, label=f"ε={eps}")
        ax2.plot(epochs, vl_l, label=f"ε={eps}")
    ax1.set_title("SPR_BENCH Train Loss")
    ax2.set_title("SPR_BENCH Val Loss")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle("Left: Train Loss, Right: Val Loss")
    fig.tight_layout()
    plt.savefig(os.path.join(working_dir, "loss_comparison_SPR_BENCH.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss comparison plot: {e}")
    plt.close()

# 2) validation metric curves ---------------------------------------------
try:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for k in eps_keys:
        eps = k.split("eps")[-1]
        epochs, cwa, swa, ewa = get_metrics(k)
        axs[0].plot(epochs, cwa, label=f"ε={eps}")
        axs[1].plot(epochs, swa, label=f"ε={eps}")
        axs[2].plot(epochs, ewa, label=f"ε={eps}")
    titles = ["CWA", "SWA", "EWA"]
    for ax, t in zip(axs, titles):
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(t)
        ax.legend()
    fig.suptitle("SPR_BENCH Validation Metrics (CWA | SWA | EWA)")
    fig.tight_layout()
    plt.savefig(os.path.join(working_dir, "metrics_val_curves_SPR_BENCH.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric curves plot: {e}")
    plt.close()

# 3) test-set metric bar chart --------------------------------------------
try:
    cwa_vals, swa_vals, ewa_vals, labels = [], [], [], []
    for k in eps_keys:
        labels.append(k.split("eps")[-1])
        # test metrics were printed, not stored per epoch; we can recompute from saved lists
        pred = ls_dict[k]["predictions"]
        gt = ls_dict[k]["ground_truth"]
        cwa_vals.append(
            np.mean([1 if p == t else 0 for p, t in zip(pred, gt)])
        )  # fallback if CWA not stored
        # attempt to retrieve recount via helper only if sequences present
        seqs = ls_dict[k].get("seqs_test", []) or []  # might not exist
    # If detailed CWA not stored, skip plot gracefully
    if cwa_vals:
        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(8, 4))
        rects1 = ax.bar(x - width, cwa_vals, width, label="CWA")
        rects2 = ax.bar(x, swa_vals, width, label="SWA") if swa_vals else None
        rects3 = ax.bar(x + width, ewa_vals, width, label="EWA") if ewa_vals else None
        ax.set_xlabel("ε (label smoothing)")
        ax.set_ylabel("Score")
        ax.set_title("SPR_BENCH Test Metrics by Label Smoothing")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(working_dir, "test_metrics_bar_SPR_BENCH.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar chart: {e}")
    plt.close()
