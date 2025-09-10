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
    experiment_data = {}

bench = experiment_data.get("weight_decay_tuning", {}).get("SPR_BENCH", {})

# shortcut: map decay value -> logs dict
decays = sorted(
    bench.keys(), key=lambda x: float(x.split("_")[1])
)  # e.g. ['decay_0.0', ...]
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# ---------- Figure 1: loss curves ----------
try:
    plt.figure(figsize=(6, 4))
    for idx, key in enumerate(decays):
        d = bench[key]
        tr = d["losses"]["train"]
        vl = d["losses"]["val"]
        epochs = np.arange(1, len(tr) + 1)
        col = colors[idx % len(colors)]
        plt.plot(epochs, tr, label=f"{key}-train", color=col, linestyle="-")
        plt.plot(epochs, vl, label=f"{key}-val", color=col, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training & Validation Loss Curves\nWeight-Decay Sweep")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_weight_decay.png")
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- Figure 2: HWA curves ----------
try:
    plt.figure(figsize=(6, 4))
    for idx, key in enumerate(decays):
        hwa = bench[key]["metrics"]["val"]
        epochs = np.arange(1, len(hwa) + 1)
        plt.plot(epochs, hwa, label=key, color=colors[idx % len(colors)])
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: Validation HWA Across Epochs\nWeight-Decay Sweep")
    plt.legend(fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves_weight_decay.png")
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# ---------- Figure 3: final HWA bar chart ----------
try:
    plt.figure(figsize=(5, 3))
    finals = [bench[k]["metrics"]["val"][-1] for k in decays]
    x = np.arange(len(decays))
    plt.bar(x, finals, color=colors[: len(decays)])
    plt.xticks(x, decays, rotation=45, ha="right")
    plt.ylabel("Final Epoch HWA")
    plt.title("SPR_BENCH: Final HWA vs Weight Decay")
    plt.tight_layout()
    fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar_weight_decay.png")
    plt.savefig(fname, dpi=150)
    print("Saved", fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()
