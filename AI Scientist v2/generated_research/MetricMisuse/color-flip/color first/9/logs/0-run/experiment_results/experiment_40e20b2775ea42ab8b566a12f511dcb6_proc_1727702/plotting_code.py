import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------------- load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

root = experiment_data.get("embedding_dim_tuning", {}).get("SPR_BENCH", {})
tags = sorted(root.keys(), key=lambda t: int(t.split("_")[-1]) if "_" in t else t)

# ----------------- figure 1: loss curves -----------------
try:
    plt.figure()
    for tag in tags:
        tr = np.array(root[tag]["losses"]["train"])
        val = np.array(root[tag]["losses"]["val"])
        plt.plot(tr[:, 0], tr[:, 1], label=f"{tag}-train")
        plt.plot(val[:, 0], val[:, 1], "--", label=f"{tag}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Training vs. Validation Loss\n(Embedding Dimension Sweep)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ----------------- figure 2: validation DWHS -----------------
try:
    plt.figure()
    for tag in tags:
        val_metrics = np.array(root[tag]["metrics"]["val"])
        epochs = val_metrics[:, 0]
        dwhs = val_metrics[:, 3]
        plt.plot(epochs, dwhs, label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("DWHS")
    plt.title("SPR_BENCH – Validation DWHS Across Epochs\n(Embedding Dimension Sweep)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_dwhs_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation DWHS curves: {e}")
    plt.close()

# ----------------- figure 3: test DWHS bar chart -----------------
try:
    plt.figure()
    dims = [int(t.split("_")[-1]) for t in tags]
    scores = [root[t]["metrics"]["test"][2] for t in tags]  # DWHS index 2
    plt.bar([str(d) for d in dims], scores)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("DWHS")
    plt.title("SPR_BENCH – Test DWHS by Embedding Dimension")
    fname = os.path.join(working_dir, "SPR_BENCH_test_dwhs_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test DWHS bar chart: {e}")
    plt.close()

# ----------------- print summary -----------------
print("Embedding Dim | Test DWHS")
for dim, sc in zip(dims, scores):
    print(f"{dim:>13} | {sc:.3f}")
