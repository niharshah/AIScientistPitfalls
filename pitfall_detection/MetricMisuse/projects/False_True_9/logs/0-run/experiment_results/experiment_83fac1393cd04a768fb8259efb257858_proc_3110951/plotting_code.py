import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

model_key = "Unidirectional_LSTM"
ds_key = "SPR_BENCH"
hidden_sizes = sorted(experiment_data.get(model_key, {}).keys())

# containers for final numbers
final_dwa = {}
final_hwa = {}

# ---------------- plot 1 : loss curves ----------------
try:
    plt.figure()
    for hs in hidden_sizes:
        store = experiment_data[model_key][hs][ds_key]
        tr = np.array(store["losses"]["train"])
        vl = np.array(store["losses"]["val"])
        plt.plot(tr[:, 0], tr[:, 1], label=f"train h={hs}")
        plt.plot(vl[:, 0], vl[:, 1], "--", label=f"val h={hs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        "SPR_BENCH: Training vs Validation Loss\nUniLSTM (Left: Train, Right: Val curves)"
    )
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------------- plot 2 : harmonic-weighted accuracy ----------------
try:
    plt.figure()
    for hs in hidden_sizes:
        store = experiment_data[model_key][hs][ds_key]
        met = np.array(store["metrics"]["val"])  # ep, swa, cwa, hwa, dwa
        plt.plot(met[:, 0], met[:, 3], label=f"h={hs}")
        # collect finals
        final_hwa[hs] = met[-1, 3]
        final_dwa[hs] = met[-1, 4]
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: Harmonic-Weighted Accuracy over Epochs\nUniLSTM")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_harmonic_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

# ---------------- plot 3 : final difficulty-weighted accuracy ----------------
try:
    plt.figure()
    xs = np.arange(len(hidden_sizes))
    vals = [final_dwa[hs] for hs in hidden_sizes]
    plt.bar(xs, vals, tick_label=[str(hs) for hs in hidden_sizes])
    plt.xlabel("Hidden Size")
    plt.ylabel("Final Difficulty-Weighted Accuracy")
    plt.title("SPR_BENCH: Final Difficulty-Weighted Accuracy per Hidden Size\nUniLSTM")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_dwa.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final DWA bar chart: {e}")
    plt.close()

# ---------------- print metrics ----------------
for hs in hidden_sizes:
    print(
        f"Hidden {hs}: final HWA={final_hwa.get(hs, np.nan):.4f}, "
        f"final DWA={final_dwa.get(hs, np.nan):.4f}"
    )
