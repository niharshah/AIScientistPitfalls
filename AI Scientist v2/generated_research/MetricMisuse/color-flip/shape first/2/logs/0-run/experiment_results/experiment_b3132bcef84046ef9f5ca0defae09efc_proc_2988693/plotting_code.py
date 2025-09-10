import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

runs = list(experiment_data.keys())

# ---------- prepare containers ----------
loss_curves = {}
val_loss_curves = {}
hwa_curves = {}
final_hwa = {}
best_hwa = {}

for run in runs:
    run_data = experiment_data[run]["SPR_BENCH"]
    loss_curves[run] = run_data["losses"]["train"]
    val_loss_curves[run] = run_data["losses"]["val"]
    hwa_curves[run] = [m["hwa"] for m in run_data["metrics"]["val"]]
    final_hwa[run] = hwa_curves[run][-1] if hwa_curves[run] else 0.0
    best_hwa[run] = max(hwa_curves[run]) if hwa_curves[run] else 0.0

# ---------- plot 1: train/val loss ----------
try:
    plt.figure()
    for run in runs:
        epochs = np.arange(1, len(loss_curves[run]) + 1)
        plt.plot(epochs, loss_curves[run], label=f"{run}-train")
        plt.plot(epochs, val_loss_curves[run], "--", label=f"{run}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Train vs Validation Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- plot 2: HWA curves ----------
try:
    plt.figure()
    for run in runs:
        epochs = np.arange(1, len(hwa_curves[run]) + 1)
        plt.plot(epochs, hwa_curves[run], label=run)
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR_BENCH: Validation Harmonic Weighted Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_hwa_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# ---------- plot 3: final HWA bar chart ----------
try:
    plt.figure()
    runs_sorted = sorted(final_hwa.keys())
    vals = [final_hwa[r] for r in runs_sorted]
    plt.bar(runs_sorted, vals)
    plt.ylabel("Final HWA")
    plt.title("SPR_BENCH: Final Epoch HWA per Run")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar chart: {e}")
    plt.close()

# ---------- print summary ----------
print("\n=== HWA SUMMARY ===")
for run in runs:
    print(f"{run}: best HWA={best_hwa[run]:.4f} | final HWA={final_hwa[run]:.4f}")
