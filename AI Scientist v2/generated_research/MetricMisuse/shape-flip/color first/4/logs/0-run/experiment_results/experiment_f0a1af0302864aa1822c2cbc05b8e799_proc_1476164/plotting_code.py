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

if experiment_data:
    emb_runs = experiment_data.get("embedding_dim", {})
    emb_keys = sorted(emb_runs.keys(), key=lambda s: int(s.split("_")[-1]))

    # ---------- Plot 1: Loss curves ----------
    try:
        plt.figure(figsize=(6, 4))
        for k in emb_keys:
            ep = np.arange(1, len(emb_runs[k]["losses"]["train"]) + 1)
            plt.plot(
                ep, emb_runs[k]["losses"]["train"], label=f"{k} train", linestyle="--"
            )
            plt.plot(ep, emb_runs[k]["losses"]["val"], label=f"{k} val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH (or synthetic) – Train vs Val Loss")
        plt.legend(fontsize=6)
        path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ---------- Plot 2: Validation CWA2 curves ----------
    try:
        plt.figure(figsize=(6, 4))
        for k in emb_keys:
            ep = np.arange(1, len(emb_runs[k]["metrics"]["val"]) + 1)
            plt.plot(ep, emb_runs[k]["metrics"]["val"], label=k)
        plt.xlabel("Epoch")
        plt.ylabel("CWA2")
        plt.title("SPR_BENCH – Validation CWA2 across Embedding Sizes")
        plt.legend(fontsize=6)
        path = os.path.join(working_dir, "SPR_BENCH_CWA2_curves.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA2 curve plot: {e}")
        plt.close()

    # ---------- Plot 3: Best CWA2 bar chart ----------
    try:
        best_vals = [max(emb_runs[k]["metrics"]["val"]) for k in emb_keys]
        plt.figure(figsize=(5, 3))
        plt.bar(emb_keys, best_vals, color="skyblue")
        plt.ylabel("Best CWA2")
        plt.title("SPR_BENCH – Best Validation CWA2 per Embedding Size")
        plt.tight_layout()
        path = os.path.join(working_dir, "SPR_BENCH_best_CWA2_bar.png")
        plt.savefig(path)
        plt.close()
    except Exception as e:
        print(f"Error creating best CWA2 bar plot: {e}")
        plt.close()
else:
    print("No experiment data found; skipping plots.")
