import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

wd_data = experiment_data.get("weight_decay", {}).get("SPR_BENCH", {})
weight_decays = sorted(float(k.split("_")[1]) for k in wd_data)

final_hwa = {}  # store for summary print

# --------------- 1) loss curves -------------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for run_key, run in wd_data.items():
        epochs, tr_loss = zip(*run["losses"]["train"])
        _, va_loss = zip(*run["losses"]["val"])
        label = run_key.split("_")[1]
        axes[0].plot(epochs, tr_loss, label=f"wd={label}")
        axes[1].plot(epochs, va_loss, label=f"wd={label}")
    axes[0].set_title("Train Loss")
    axes[1].set_title("Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
    fig.suptitle("SPR_BENCH Loss Curves (Left: Train, Right: Val)")
    plt.tight_layout()
    path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# --------------- 2) HWA curves --------------------------
try:
    plt.figure(figsize=(5, 4))
    for run_key, run in wd_data.items():
        epochs, swa, cwa, hwa = zip(*run["metrics"]["val"])
        label = run_key.split("_")[1]
        plt.plot(epochs, hwa, marker="o", label=f"wd={label}")
        final_hwa[label] = hwa[-1]
    plt.title("SPR_BENCH Harmonic Weighted Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
    plt.savefig(path)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# --------------- 3) final HWA vs weight_decay -----------
try:
    labels = sorted(final_hwa.keys(), key=lambda x: float(x))
    values = [final_hwa[k] for k in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color="skyblue")
    plt.title("SPR_BENCH Final Epoch HWA vs Weight Decay")
    plt.xlabel("Weight Decay")
    plt.ylabel("Final HWA")
    plt.tight_layout()
    path = os.path.join(working_dir, "SPR_BENCH_final_hwa_vs_wd.png")
    plt.savefig(path)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar chart: {e}")
    plt.close()

# ------------------- summary print ----------------------
if final_hwa:
    print("Final epoch HWA per weight decay:")
    for wd in labels:
        print(f"  weight_decay={wd}: HWA={final_hwa[wd]:.4f}")
