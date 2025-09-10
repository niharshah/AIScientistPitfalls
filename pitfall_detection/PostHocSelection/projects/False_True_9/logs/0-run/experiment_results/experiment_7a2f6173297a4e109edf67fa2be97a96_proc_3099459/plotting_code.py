import matplotlib.pyplot as plt
import numpy as np
import os

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

clip_dict = experiment_data.get("gradient_clipping_max_norm", {})


# helper to pull curves
def curve(vals, idx):
    return [v[idx] for v in vals]  # idx: 1 for loss, 1/2/3 for metrics


colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
clip_keys = sorted(clip_dict.keys(), key=float)[:5]  # at most 5

# ---------- 1) loss curves ----------
try:
    plt.figure(figsize=(6, 4))
    for c, key in zip(colors, clip_keys):
        epochs, tr_loss = zip(*clip_dict[key]["losses"]["train"])
        _, val_loss = zip(*clip_dict[key]["losses"]["val"])
        plt.plot(epochs, tr_loss, c=c, ls="-", label=f"train clip={key}")
        plt.plot(epochs, val_loss, c=c, ls="--", label=f"val clip={key}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "SPR_BENCH – Loss vs. Epochs\nLeft: Training (solid), Right: Validation (dashed)"
    )
    plt.legend(fontsize=7)
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- 2) validation HWA curves ----------
try:
    plt.figure(figsize=(6, 4))
    for c, key in zip(colors, clip_keys):
        epochs = [v[0] for v in clip_dict[key]["metrics"]["val"]]
        hwa = [v[3] for v in clip_dict[key]["metrics"]["val"]]
        plt.plot(epochs, hwa, c=c, label=f"clip={key}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Acc")
    plt.title("SPR_BENCH – Validation HWA Across Epochs")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_hwa_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating HWA curves: {e}")
    plt.close()

# ---------- 3) final-epoch HWA bar chart ----------
final_hwa = []
for key in clip_keys:
    final_hwa.append(clip_dict[key]["metrics"]["val"][-1][3])

try:
    plt.figure(figsize=(5, 3))
    plt.bar(clip_keys, final_hwa, color=colors[: len(clip_keys)])
    plt.xlabel("Gradient Clip Max-Norm")
    plt.ylabel("Final Epoch HWA")
    plt.title("SPR_BENCH – Final Harmonic Weighted Accuracy by Clip Value")
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_final_hwa_bar.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating final HWA bar: {e}")
    plt.close()

# ---------- print summary ----------
print("Final-epoch HWA:")
for k, h in zip(clip_keys, final_hwa):
    print(f"  clip={k:>4}: HWA={h:.4f}")
