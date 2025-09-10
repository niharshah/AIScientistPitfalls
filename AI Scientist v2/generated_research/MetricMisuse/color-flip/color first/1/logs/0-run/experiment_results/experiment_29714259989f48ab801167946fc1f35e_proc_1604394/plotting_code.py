import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
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

dataset_name = "SPR_BENCH"
tags = sorted(experiment_data.keys())  # e.g. ['batch_size_32', ...]

# ---------- gather all arrays ----------
loss_train = {}
loss_val = {}
hmwa_val = {}
test_metrics = {}

for tag in tags:
    ed = experiment_data[tag][dataset_name]
    loss_train[tag] = ed["losses"]["train"]
    loss_val[tag] = ed["losses"]["val"]
    hmwa_val[tag] = [m["hmwa"] for m in ed["metrics"]["val"]]
    test_metrics[tag] = ed["metrics"]["test"]

# ---------- figure 1: loss curves ----------
try:
    plt.figure(figsize=(6, 4))
    for tag in tags:
        plt.plot(loss_train[tag], label=f"{tag}-train")
        plt.plot(loss_val[tag], linestyle="--", label=f"{tag}-val")
    plt.title(f"{dataset_name} Loss Curves\nSolid: Train, Dashed: Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, f"{dataset_name}_loss_curves.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# ---------- figure 2: HMWA curves ----------
try:
    plt.figure(figsize=(6, 4))
    for tag in tags:
        plt.plot(hmwa_val[tag], label=tag)
    plt.title(f"{dataset_name} Validation HMWA per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("HMWA")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, f"{dataset_name}_hmwa_curves.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating HMWA figure: {e}")
    plt.close()

# ---------- figure 3: test metrics ----------
try:
    metrics = ["cwa", "swa", "hmwa"]
    x = np.arange(len(tags))
    width = 0.25
    plt.figure(figsize=(7, 4))
    for i, m in enumerate(metrics):
        vals = [test_metrics[t][m] for t in tags]
        plt.bar(x + i * width - width, vals, width, label=m.upper())
    plt.xticks(x, tags, rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title(f"{dataset_name} Test Metrics")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset_name}_test_metrics.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating test metrics figure: {e}")
    plt.close()

# ---------- print numeric test metrics ----------
print("=== Test Metrics ===")
for tag in tags:
    tm = test_metrics[tag]
    print(f"{tag}: CWA={tm['cwa']:.4f} | SWA={tm['swa']:.4f} | HMWA={tm['hmwa']:.4f}")
