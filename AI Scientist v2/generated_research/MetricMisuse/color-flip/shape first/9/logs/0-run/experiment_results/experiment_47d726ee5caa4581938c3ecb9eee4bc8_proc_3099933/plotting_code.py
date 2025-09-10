import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr_data = experiment_data.get("embedding_dim", {}).get("SPR_BENCH", {})
if not spr_data:
    print("No SPR_BENCH data found, nothing to plot.")
    exit()

dims = sorted(int(k.split("_")[-1]) for k in spr_data.keys())
loss_train, loss_val, cwa_val = {}, {}, {}
for d in dims:
    entry = spr_data[f"dim_{d}"]
    loss_train[d] = entry["losses"]["train"]
    loss_val[d] = entry["losses"]["val"]
    cwa_val[d] = entry["metrics"]["val"]

# ----------------- figure 1: loss curves -----------------
try:
    plt.figure()
    for d in dims:
        epochs = np.arange(1, len(loss_train[d]) + 1)
        plt.plot(epochs, loss_train[d], "--", label=f"Train dim={d}")
        plt.plot(epochs, loss_val[d], "-", label=f"Val   dim={d}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------------- figure 2: CWA-2D curves -----------------
try:
    plt.figure()
    for d in dims:
        epochs = np.arange(1, len(cwa_val[d]) + 1)
        plt.plot(epochs, cwa_val[d], label=f"dim={d}")
    plt.xlabel("Epoch")
    plt.ylabel("CWA-2D")
    plt.title("SPR_BENCH Complexity-Weighted Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_cwa_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWA curve plot: {e}")
    plt.close()

# ---------------- figure 3: final-epoch CWA summary -------
try:
    plt.figure()
    final_scores = [cwa_val[d][-1] for d in dims]
    plt.bar([str(d) for d in dims], final_scores, color="skyblue")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Final Epoch CWA-2D")
    plt.title("SPR_BENCH Final Complexity-Weighted Accuracy by Embedding Size")
    fname = os.path.join(working_dir, "spr_bench_final_cwa_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CWA bar plot: {e}")
    plt.close()

# ------------- print evaluation metrics -------------
print("Final-epoch CWA-2D scores:")
for d, score in zip(dims, final_scores):
    print(f"  dim={d}: {score:.4f}")
