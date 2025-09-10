import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to safely fetch
def safe_get(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


dataset = "SPR_BENCH"
embed_dict = safe_get(experiment_data, "embedding_dim", dataset, default={})
dims = (
    sorted(embed_dict.keys(), key=lambda s: int(s.split("_")[-1])) if embed_dict else []
)

# ---- Figure 1: train/val loss curves ---- #
try:
    plt.figure()
    for dim in dims:
        tr_losses = embed_dict[dim]["losses"]["train"]
        val_losses = embed_dict[dim]["losses"]["val"]
        epochs = range(1, len(tr_losses) + 1)
        plt.plot(epochs, tr_losses, label=f"{dim}-train")
        plt.plot(epochs, val_losses, linestyle="--", label=f"{dim}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"Loss Curves - {dataset}")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset}_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---- Figure 2: per-epoch validation GCWA ---- #
try:
    plt.figure()
    for dim in dims:
        val_metrics = embed_dict[dim]["metrics"]["val"]
        gcwa = [m["GCWA"] for m in val_metrics]
        epochs = range(1, len(gcwa) + 1)
        plt.plot(epochs, gcwa, label=dim)
    plt.xlabel("Epoch")
    plt.ylabel("GCWA")
    plt.title(f"Validation GCWA over Epochs - {dataset}")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset}_val_GCWA.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating GCWA plot: {e}")
    plt.close()

# ---- Figure 3: final test metrics vs embedding dim ---- #
try:
    metrics_names = ["CWA", "SWA", "GCWA"]
    x = np.arange(len(dims))
    width = 0.25
    plt.figure()
    for i, m in enumerate(metrics_names):
        vals = [embed_dict[dim]["metrics"]["test"][m] for dim in dims]
        plt.bar(x + i * width, vals, width=width, label=m)
    plt.xticks(x + width, [dim for dim in dims])
    plt.ylabel("Score")
    plt.title(f"Test Metrics vs Embedding Dim - {dataset}")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset}_test_metrics_bar.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar plot: {e}")
    plt.close()
