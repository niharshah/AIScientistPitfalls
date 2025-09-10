import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- io ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

hidden_dims = ["64", "128", "256"]
loss_dict = {}
dwa_dict = {}

# ---------- gather ----------
for h in hidden_dims:
    try:
        entry = experiment_data["gcn_hidden_dim"]["SPR_BENCH"][h]
        tr_losses = [v for _, v in entry["losses"]["train"]]
        val_losses = [v for _, v in entry["losses"]["val"]]
        val_dwa = [v for _, v in entry["metrics"]["val"]]
        loss_dict[h] = (tr_losses, val_losses)
        dwa_dict[h] = val_dwa
    except Exception as e:
        print(f"Hidden dim {h} missing: {e}")

# ---------- plot 1: loss curves ----------
try:
    plt.figure()
    for h, (tr, val) in loss_dict.items():
        epochs = range(1, len(tr) + 1)
        plt.plot(epochs, tr, "--", label=f"train h={h}")
        plt.plot(epochs, val, "-", label=f"val h={h}")
    plt.title("SPR_BENCH Loss Curves (GCN)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- plot 2: validation DWA ----------
try:
    plt.figure()
    for h, vals in dwa_dict.items():
        epochs = range(1, len(vals) + 1)
        plt.plot(epochs, vals, label=f"h={h}")
    plt.title("SPR_BENCH Validation Dual Weighted Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("DWA")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_DWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating DWA plot: {e}")
    plt.close()

# ---------- print summary ----------
final_scores = []
for h, vals in dwa_dict.items():
    if vals:
        final_scores.append((h, vals[-1]))
        print(f"Final DWA for hidden_dim={h}: {vals[-1]:.4f}")
if final_scores:
    best_h, best_score = max(final_scores, key=lambda x: x[1])
    print(f"Best hidden_dim is {best_h} with DWA={best_score:.4f}")
