import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up ----------
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

bs_section = experiment_data.get("batch_size", {})
bs_keys = sorted(bs_section.keys(), key=int)

# containers for summary printout
summary = []

# ---------- figure 1: training loss ----------
try:
    plt.figure()
    for bs in bs_keys:
        epochs, tr_loss = zip(*bs_section[bs]["losses"]["train"])
        plt.plot(epochs, tr_loss, marker="o", label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("SPR_BENCH: Training Loss over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_training_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# ---------- figure 2: validation loss ----------
try:
    plt.figure()
    for bs in bs_keys:
        epochs, val_loss = zip(*bs_section[bs]["losses"]["val"])
        plt.plot(epochs, val_loss, marker="o", label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("SPR_BENCH: Validation Loss over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_validation_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()

# ---------- figure 3: validation HWA ----------
try:
    plt.figure()
    for bs in bs_keys:
        epochs, swa, cwa, hwa = zip(*bs_section[bs]["metrics"]["val"])
        plt.plot(epochs, hwa, marker="o", label=f"bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title("SPR_BENCH: Validation Harmonic Weighted Accuracy over Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_validation_hwa.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA plot: {e}")
    plt.close()

# ---------- figure 4: final HWA bar ----------
try:
    plt.figure()
    final_hwa = []
    for bs in bs_keys:
        *_, last = bs_section[bs]["metrics"]["val"]
        final_hwa.append(last[3])  # last element is hwa
        summary.append((bs, last[1], last[2], last[3]))  # swa, cwa, hwa
    x = np.arange(len(bs_keys))
    plt.bar(x, final_hwa, tick_label=bs_keys)
    plt.xlabel("Batch Size")
    plt.ylabel("Final Epoch HWA")
    plt.title("SPR_BENCH: Final Harmonic Weighted Accuracy by Batch Size")
    fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

# ---------- print summary ----------
print("Final Validation Metrics per Batch Size")
print(f"{'Batch':>6} | {'SWA':>6} | {'CWA':>6} | {'HWA':>6}")
print("-" * 30)
for bs, swa, cwa, hwa in sorted(summary, key=lambda x: int(x[0])):
    print(f"{bs:>6} | {swa:6.3f} | {cwa:6.3f} | {hwa:6.3f}")
