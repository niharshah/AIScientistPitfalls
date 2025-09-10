import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

hidden_dict = experiment_data.get("hidden_size", {})
if not hidden_dict:
    print("No hidden_size data found.")
    exit()

# -------------- collect statistics ----------
losses = {}  # {hs: {'train': (ep, val), 'val': (ep, val)}}
metrics_hwa = {}  # {hs: [(ep, hwa)]}
final_hwa = {}  # {hs: hwa}
for hs, result in hidden_dict.items():
    rec = result.get("SPR_BENCH", {})
    tr_loss = rec.get("losses", {}).get("train", [])
    val_loss = rec.get("losses", {}).get("val", [])
    hwa = [(e, h) for e, _, _, h in rec.get("metrics", {}).get("val", [])]
    if tr_loss and val_loss and hwa:
        losses[hs] = {"train": tr_loss, "val": val_loss}
        metrics_hwa[hs] = hwa
        final_hwa[hs] = hwa[-1][1]

# ---------------- plot losses ---------------
try:
    plt.figure()
    for hs, lv in sorted(losses.items()):
        ep_t, val_t = zip(*lv["train"])
        ep_v, val_v = zip(*lv["val"])
        plt.plot(ep_t, val_t, label=f"train hs={hs}")
        plt.plot(ep_v, val_v, linestyle="--", label=f"val hs={hs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss (Hidden-Size Sweep)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves_hidden_sizes.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------------- plot HWA curves -----------
try:
    plt.figure()
    for hs, arr in sorted(metrics_hwa.items()):
        ep, hwa = zip(*arr)
        plt.plot(ep, hwa, label=f"hs={hs}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Accuracy")
    plt.title("SPR_BENCH: HWA Curves Across Hidden Sizes")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves_hidden_sizes.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# --------------- bar chart final HWA -------
try:
    plt.figure()
    h_sizes, h_vals = zip(*sorted(final_hwa.items()))
    plt.bar([str(h) for h in h_sizes], h_vals, color="skyblue")
    plt.xlabel("Hidden Size")
    plt.ylabel("Final-Epoch HWA")
    plt.title("SPR_BENCH: Final Harmonic Weighted Accuracy by Hidden Size")
    fname = os.path.join(working_dir, "SPR_BENCH_final_hwa_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar chart: {e}")
    plt.close()

# --------------- print summary --------------
print("Final-epoch HWA per hidden size:")
for hs, hwa in sorted(final_hwa.items()):
    print(f"  hidden={hs:>3}: HWA={hwa:.4f}")
