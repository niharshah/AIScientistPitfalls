import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- setup ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    emb_dict = experiment_data.get("embedding_dim", {})
    emb_dims = sorted(emb_dict.keys())

    # Helper to pull series
    def series(emb, part, key):
        if part == "losses":
            return emb_dict[emb]["losses"][key]  # key='train'/'val'
        elif part == "metrics":
            return [d[key] for d in emb_dict[emb]["metrics"]["val"]]
        else:
            return []

    # --------- 1. Training loss curves ----------
    try:
        plt.figure()
        for emb in emb_dims:
            plt.plot(series(emb, "losses", "train"), label=f"emb={emb}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training Loss vs Epoch")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_train_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    # --------- 2. Validation loss curves ----------
    try:
        plt.figure()
        for emb in emb_dims:
            plt.plot(series(emb, "losses", "val"), label=f"emb={emb}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Validation Loss vs Epoch")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_loss_curves.png")
        plt.savefig(fname)
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating validation loss plot: {e}")
        plt.close()

    # --------- 3. Validation HWA curves ----------
    try:
        plt.figure()
        for emb in emb_dims:
            plt.plot(series(emb, "metrics", "hwa"), label=f"emb={emb}")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.title("SPR_BENCH: Harmonic Weighted Accuracy vs Epoch")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_hwa_curves.png")
        plt.savefig(fname)
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # --------- 4. Best HWA bar chart ----------
    try:
        best_hwa = [max(series(emb, "metrics", "hwa")) for emb in emb_dims]
        plt.figure()
        plt.bar([str(e) for e in emb_dims], best_hwa)
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Best HWA")
        plt.title("SPR_BENCH: Best Validation HWA per Embedding Size")
        fname = os.path.join(working_dir, "SPR_BENCH_best_hwa_bar.png")
        plt.savefig(fname)
        plt.close()
        print("Saved", fname)
    except Exception as e:
        print(f"Error creating best HWA bar chart: {e}")
        plt.close()
