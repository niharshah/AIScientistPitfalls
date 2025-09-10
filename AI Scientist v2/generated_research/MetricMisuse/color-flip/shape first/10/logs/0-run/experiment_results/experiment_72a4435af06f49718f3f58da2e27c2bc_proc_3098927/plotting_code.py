import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("batch_size_sweep", {}).get("SPR_BENCH", {})

# 1) Train / Val loss curves for each batch size
try:
    plt.figure()
    for bs, store in spr_data.items():
        epochs = np.arange(1, len(store["metrics"]["train_loss"]) + 1)
        plt.plot(epochs, store["metrics"]["train_loss"], label=f"train bs={bs}")
        plt.plot(epochs, store["metrics"]["val_loss"], "--", label=f"val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Train vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# 2) Final HWA per batch size
try:
    plt.figure()
    bss, hwas = [], []
    for bs, store in spr_data.items():
        bss.append(int(bs))
        hwas.append(store["metrics"]["HWA"][-1] if store["metrics"]["HWA"] else 0)
    plt.bar(bss, hwas)
    plt.xlabel("Batch Size")
    plt.ylabel("Final Epoch HWA")
    plt.title("SPR_BENCH: Final HWA vs Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_HWA_by_batch_size.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA bar plot: {e}")
    plt.close()

# 3) SWA & CWA across epochs for batch size 32 (if present)
try:
    if "32" in spr_data:
        store = spr_data["32"]
        epochs = np.arange(1, len(store["metrics"]["SWA"]) + 1)
        plt.figure()
        plt.plot(epochs, store["metrics"]["SWA"], label="SWA")
        plt.plot(epochs, store["metrics"]["CWA"], label="CWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH: SWA & CWA over Epochs (bs=32)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_SWA_CWA_bs32.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating SWA/CWA plot: {e}")
    plt.close()
