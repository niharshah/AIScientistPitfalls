import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- setup ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("num_gnn_layers", {}).get("SPR_BENCH", {})
if not spr_data:
    print("No SPR_BENCH data found. Exiting.")
    quit()

layers = sorted(spr_data.keys())
epochs = list(range(1, len(next(iter(spr_data.values()))["losses"]["train"]) + 1))

# ---------------- plot 1: loss curves ----------------
try:
    plt.figure()
    for l in layers:
        plt.plot(epochs, spr_data[l]["losses"]["train"], label=f"Train L{l}")
        plt.plot(
            epochs, spr_data[l]["losses"]["val"], linestyle="--", label=f"Val L{l}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training and Validation Loss vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------------- plot 2: validation HWA curves ----------------
try:
    plt.figure()
    for l in layers:
        hwa = [m["hwa"] for m in spr_data[l]["metrics"]["val"]]
        plt.plot(epochs, hwa, label=f"L{l}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: Validation HWA vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_HWA.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

# ---------------- plot 3: final test HWA ----------------
try:
    plt.figure()
    hwa_test = [spr_data[l]["metrics"]["test"]["hwa"] for l in layers]
    plt.bar([str(l) for l in layers], hwa_test)
    plt.xlabel("# GraphSAGE Layers")
    plt.ylabel("Test HWA")
    plt.title("SPR_BENCH: Test HWA by # Layers")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_HWA_by_layers.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test HWA bar plot: {e}")
    plt.close()

# ---------------- print final metrics ----------------
for l in layers:
    tm = spr_data[l]["metrics"]["test"]
    print(
        f"Layers {l} - Test CWA: {tm['cwa']:.3f} | SWA: {tm['swa']:.3f} | HWA: {tm['hwa']:.3f}"
    )
