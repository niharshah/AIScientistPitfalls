import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = experiment_data.get("weight_decay", {}).get("SPR_BENCH", {}).get("runs", [])

# Organize data
wds, train_losses, val_losses, hwas = [], [], [], []
for run in runs:
    wds.append(run["weight_decay"])
    train_losses.append(run["losses"]["train"])
    val_losses.append(run["losses"]["val"])
    hwas.append(run["metrics"]["HWA"])

# ---------- Plot 1: Loss curves ----------
try:
    plt.figure()
    epochs = np.arange(1, len(train_losses[0]) + 1)
    for wd, tr, vl in zip(wds, train_losses, val_losses):
        plt.plot(epochs, tr, label=f"train wd={wd}")
        plt.plot(epochs, vl, "--", label=f"val wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves vs Weight Decay")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_weight_decay.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- Plot 2: HWA curves ----------
try:
    plt.figure()
    epochs = np.arange(1, len(hwas[0]) + 1)
    for wd, hw in zip(wds, hwas):
        plt.plot(epochs, hw, label=f"HWA wd={wd}")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR_BENCH HWA Across Epochs vs Weight Decay")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_HWA_curves_weight_decay.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ---------- Plot 3: Final HWA vs WD ----------
try:
    plt.figure()
    final_hwa = [hw[-1] for hw in hwas]
    plt.scatter(wds, final_hwa)
    for wd, hw in zip(wds, final_hwa):
        plt.annotate(f"{hw:.2f}", (wd, hw))
    plt.xscale("log")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final HWA")
    plt.title("SPR_BENCH Final HWA vs Weight Decay")
    fname = os.path.join(working_dir, "SPR_BENCH_final_HWA_vs_weight_decay.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA scatter plot: {e}")
    plt.close()

# ---------- Report best configuration ----------
if hwas:
    best_idx = int(np.argmax([hw[-1] for hw in hwas]))
    print(f"Best weight_decay={wds[best_idx]} with final HWA={hwas[best_idx][-1]:.3f}")
