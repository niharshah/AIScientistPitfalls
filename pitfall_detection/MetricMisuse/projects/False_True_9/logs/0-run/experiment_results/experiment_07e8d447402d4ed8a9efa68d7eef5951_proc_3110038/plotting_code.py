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

runs = experiment_data.get("mean_pool_ablation", {}).get("SPR_BENCH", {})
emb_keys = sorted(
    [k for k in runs.keys() if k.startswith("emb_")],
    key=lambda x: int(x.split("_")[-1]),
)


# helper to unpack epoch/value tuples ------------------------------------------
def get_curve(tuples):
    epochs, vals = zip(*tuples)
    return list(epochs), list(vals)


# -------------- figure 1 : loss curves ----------------------------------------
try:
    fig, axes = plt.subplots(
        1, len(emb_keys), figsize=(5 * len(emb_keys), 4), sharey=True
    )
    if len(emb_keys) == 1:
        axes = [axes]
    for ax, emb in zip(axes, emb_keys):
        epochs_tr, loss_tr = get_curve(runs[emb]["losses"]["train"])
        epochs_v, loss_v = get_curve(runs[emb]["losses"]["val"])
        ax.plot(epochs_tr, loss_tr, label="Train Loss")
        ax.plot(epochs_v, loss_v, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title(f"Loss Curves ‑ {emb}")
        ax.legend()
    fig.suptitle(
        "SPR_BENCH Mean-Pool Ablation\nLeft: Train vs Val Loss per Embedding Dim"
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(out_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------------- figure 2 : HWA curves -----------------------------------------
try:
    plt.figure(figsize=(6, 4))
    for emb in emb_keys:
        epochs, swa, cwa, hwa = zip(*runs[emb]["metrics"]["val"])
        plt.plot(epochs, hwa, label=f"{emb}")
    plt.xlabel("Epoch")
    plt.ylabel("Harmonic Weighted Acc.")
    plt.title("SPR_BENCH: HWA vs Epoch (All Embeddings)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
    plt.savefig(out_path)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve plot: {e}")
    plt.close()

# -------------- figure 3 : final metric bar chart -----------------------------
try:
    metrics = {"SWA": [], "CWA": [], "HWA": []}
    for emb in emb_keys:
        _, swa, cwa, hwa = runs[emb]["metrics"]["val"][-1]
        metrics["SWA"].append(swa)
        metrics["CWA"].append(cwa)
        metrics["HWA"].append(hwa)
    x = np.arange(len(emb_keys))
    width = 0.25
    plt.figure(figsize=(7, 4))
    for i, (mname, vals) in enumerate(metrics.items()):
        plt.bar(x + i * width - width, vals, width, label=mname)
    plt.xticks(x, emb_keys)
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Final Epoch Weighted Accuracies")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(working_dir, "SPR_BENCH_final_metrics.png")
    plt.savefig(out_path)
    plt.close()
except Exception as e:
    print(f"Error creating final metric bar plot: {e}")
    plt.close()
