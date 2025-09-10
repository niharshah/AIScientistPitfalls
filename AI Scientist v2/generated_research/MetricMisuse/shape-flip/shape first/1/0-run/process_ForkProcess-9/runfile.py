import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# load experiment data                                               #
# ------------------------------------------------------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_metrics = experiment_data["SPR_BENCH"]["metrics"]
    spr_losses = experiment_data["SPR_BENCH"]["losses"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_metrics, spr_losses = {}, {}


# helper to get metric safely
def get(metric_dict, key):
    return metric_dict.get(key, [])


epochs = np.arange(1, len(get(spr_metrics, "train_acc")) + 1)

# ------------------------------------------------------------------ #
# 1) train & val accuracy                                            #
# ------------------------------------------------------------------ #
try:
    if len(epochs):
        plt.figure()
        plt.plot(epochs, spr_metrics["train_acc"], label="Train Acc")
        plt.plot(epochs, spr_metrics["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: Training vs Validation Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 2) train & val loss                                                #
# ------------------------------------------------------------------ #
try:
    if len(epochs):
        plt.figure()
        plt.plot(epochs, spr_losses["train"], label="Train Loss")
        plt.plot(epochs, spr_losses["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 3) Harmonic-mean accuracy (HMA)                                    #
# ------------------------------------------------------------------ #
try:
    hma = get(spr_metrics, "hma")
    if len(hma):
        plt.figure()
        plt.plot(epochs, hma, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("HMA")
        plt.title("SPR_BENCH: Harmonic-Mean Accuracy per Epoch")
        fname = os.path.join(working_dir, "SPR_BENCH_hma_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating HMA plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 4) Shape- vs Color-Weighted Accuracy                               #
# ------------------------------------------------------------------ #
try:
    swa, cwa = get(spr_metrics, "swa"), get(spr_metrics, "cwa")
    if len(swa) and len(cwa):
        plt.figure()
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH: Shape vs Color Weighted Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_weighted_accuracy.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating weighted accuracy plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# 5) Zero-Shot Accuracy                                              #
# ------------------------------------------------------------------ #
try:
    zs = get(spr_metrics, "zs_acc")
    if len(zs):
        plt.figure()
        plt.plot(epochs, zs, marker="x", color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("Zero-Shot Accuracy")
        plt.title("SPR_BENCH: Zero-Shot Accuracy per Epoch")
        fname = os.path.join(working_dir, "SPR_BENCH_zeroshot_accuracy.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating zero-shot plot: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# print final epoch summary                                          #
# ------------------------------------------------------------------ #
if len(epochs):
    idx = -1  # last epoch
    summary = {
        "Final Train Acc": spr_metrics["train_acc"][idx],
        "Final Val Acc": spr_metrics["val_acc"][idx],
        "Final Val Loss": spr_losses["val"][idx],
        "Final HMA": spr_metrics["hma"][idx],
        "Final ZS Acc": spr_metrics["zs_acc"][idx],
    }
    print("Final Epoch Metrics:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
