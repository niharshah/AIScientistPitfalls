import matplotlib.pyplot as plt
import numpy as np
import os

# set up working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(experiment_path, allow_pickle=True).item()
    spr_results = experiment_data["num_epochs_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_results = {}

# gather sweep keys sorted numerically
eps = sorted(spr_results.keys(), key=lambda x: int(x))

# helper containers
train_losses, val_losses, hwas, finals = {}, {}, {}, {}

for ep in eps:
    res = spr_results[ep]
    train_losses[ep] = res["losses"]["train"]
    val_losses[ep] = res["losses"]["val"]
    hwas[ep] = res["metrics"]["val"]
    finals[ep] = hwas[ep][-1] if hwas[ep] else None

# 1) loss curves
try:
    plt.figure()
    for ep in eps:
        x = list(range(1, len(train_losses[ep]) + 1))
        plt.plot(x, train_losses[ep], label=f"{ep}e - train")
        plt.plot(x, val_losses[ep], "--", label=f"{ep}e - val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Train vs. Validation Loss Curves")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# 2) HWA curves
try:
    plt.figure()
    for ep in eps:
        x = list(range(1, len(hwas[ep]) + 1))
        plt.plot(x, hwas[ep], label=f"{ep}e")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: Validation HWA Curves")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves plot: {e}")
    plt.close()

# 3) final HWA vs epochs
try:
    plt.figure()
    xs = [int(e) for e in eps]
    ys = [finals[e] for e in eps]
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Total Training Epochs")
    plt.ylabel("Final Validation HWA")
    plt.title("SPR_BENCH: Final HWA vs. Training Epochs")
    for x, y in zip(xs, ys):
        plt.text(x, y, f"{y:.2f}")
    save_path = os.path.join(working_dir, "SPR_BENCH_final_hwa_vs_epochs.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA plot: {e}")
    plt.close()

# print quick summary table
print("=== Final Validation HWA per Sweep ===")
for ep in eps:
    print(f"{ep} epochs : {finals[ep]:.3f}")
