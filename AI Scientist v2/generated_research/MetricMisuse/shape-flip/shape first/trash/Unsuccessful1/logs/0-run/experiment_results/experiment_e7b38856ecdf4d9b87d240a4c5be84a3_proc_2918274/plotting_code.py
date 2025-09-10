import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up paths ----------
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

bs_runs = experiment_data.get("batch_size_tuning", {})
if not bs_runs:
    print("No batch_size_tuning data found.")
    exit()


# ---------- helper to parse numbers ----------
def tag_to_bs(tag: str):
    try:
        return int(tag.split("bs")[-1])
    except Exception:
        return tag


# ---------- collect for printing ----------
final_metrics = {}

# ---------- PLOT 1: loss curves ----------
try:
    plt.figure()
    for tag, run in bs_runs.items():
        bs = tag_to_bs(tag)
        epochs = np.arange(1, len(run["losses"]["train"]) + 1)
        plt.plot(epochs, run["losses"]["train"], label=f"train bs={bs}", linestyle="--")
        plt.plot(epochs, run["losses"]["val"], label=f"val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- PLOT 2: validation HWA ----------
try:
    plt.figure()
    for tag, run in bs_runs.items():
        bs = tag_to_bs(tag)
        hwa_vals = [m["HWA"] for m in run["metrics"]["val"]]
        epochs = np.arange(1, len(hwa_vals) + 1)
        plt.plot(epochs, hwa_vals, label=f"val HWA bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Acc.")
    plt.title("SPR_BENCH: Validation HWA Across Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_HWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation HWA plot: {e}")
    plt.close()

# ---------- PLOT 3: final test HWA ----------
try:
    plt.figure()
    bsz_list, hwa_test = [], []
    for tag, run in bs_runs.items():
        bs = tag_to_bs(tag)
        test_hwa = run["metrics"]["test"]["HWA"]
        bsz_list.append(bs)
        hwa_test.append(test_hwa)
        final_metrics[bs] = test_hwa
    plt.bar([str(b) for b in bsz_list], hwa_test, color="skyblue")
    plt.xlabel("Batch Size")
    plt.ylabel("Test HWA")
    plt.title("SPR_BENCH: Test Harmonic Weighted Accuracy by Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_test_HWA_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test HWA bar chart: {e}")
    plt.close()

# ---------- print summary ----------
print("Final Test HWA by Batch Size")
for bs in sorted(final_metrics):
    print(f"  Batch Size {bs}: HWA = {final_metrics[bs]:.4f}")
