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

# ---------- parse useful pieces -----------
results = {}  # {hs: {"loss_tr":[], "loss_val":[], "hwa":[], "swa":[], "cwa":[]}}
try:
    hs_dict = experiment_data["frozen_random_embedding"]["SPR_BENCH"]["hidden_size"]
    for hs, store in hs_dict.items():
        tr = store["losses"]["train"]
        val = store["losses"]["val"]
        met = store["metrics"]["val"]
        results[hs] = {
            "loss_tr": [l for _, l in tr],
            "loss_val": [l for _, l in val],
            "hwa": [h for _, _, _, h in met],
            "swa": [s for _, s, _, _ in met],
            "cwa": [c for _, _, c, _ in met],
        }
except KeyError as e:
    print(f"Data structure missing key: {e}")

# ---------- plot 1: loss curves ------------
try:
    plt.figure()
    for hs, vals in results.items():
        epochs = np.arange(1, len(vals["loss_tr"]) + 1)
        plt.plot(epochs, vals["loss_tr"], label=f"train h={hs}")
        plt.plot(epochs, vals["loss_val"], "--", label=f"val h={hs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- plot 2: HWA curves -------------
try:
    plt.figure()
    for hs, vals in results.items():
        epochs = np.arange(1, len(vals["hwa"]) + 1)
        plt.plot(epochs, vals["hwa"], label=f"h={hs}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: Validation HWA Across Epochs")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_HWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# ---------- plot 3: final HWA bar ---------
try:
    plt.figure()
    hs_list = sorted(results.keys())
    final_hwa = [results[h]["hwa"][-1] for h in hs_list]
    plt.bar([str(h) for h in hs_list], final_hwa)
    plt.xlabel("Hidden Size")
    plt.ylabel("Final-Epoch HWA")
    plt.title("SPR_BENCH: Final Harmonic Weighted Accuracy vs Hidden Size")
    fname = os.path.join(working_dir, "SPR_BENCH_final_HWA_vs_hidden.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar plot: {e}")
    plt.close()

# ---------- print numeric summary ----------
for hs in sorted(results.keys()):
    swa = results[hs]["swa"][-1]
    cwa = results[hs]["cwa"][-1]
    hwa = results[hs]["hwa"][-1]
    print(f"Hidden={hs} | Final SWA={swa:.4f} CWA={cwa:.4f} HWA={hwa:.4f}")
