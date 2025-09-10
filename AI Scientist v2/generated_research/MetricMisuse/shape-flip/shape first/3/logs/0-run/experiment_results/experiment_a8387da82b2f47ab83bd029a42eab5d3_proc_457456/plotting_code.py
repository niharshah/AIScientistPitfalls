import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data_tree = experiment_data.get("embed_dim", {}).get("SPR_BENCH", {})
    embed_dims = sorted(data_tree.keys(), key=lambda x: int(x))

    # per-dimension plots
    for emb in embed_dims:
        dat = data_tree[emb]
        losses_tr = dat["losses"]["train"]
        losses_val = dat["losses"]["val"]
        metrics_val = dat["metrics"]["val"]  # list of dicts per epoch
        epochs = np.arange(1, len(losses_tr) + 1)

        # -------- loss curve --------
        try:
            plt.figure()
            plt.plot(epochs, losses_tr, label="Train Loss")
            plt.plot(epochs, losses_val, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"Dataset: SPR_BENCH | Train vs Val Loss (embed_dim={emb})")
            plt.legend()
            fname = f"SPR_BENCH_loss_curves_embed{emb}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for emb={emb}: {e}")
            plt.close()

        # -------- metric curve --------
        try:
            swa = [m["SWA"] for m in metrics_val]
            cwa = [m["CWA"] for m in metrics_val]
            hwa = [m["HWA"] for m in metrics_val]

            plt.figure()
            plt.plot(epochs, swa, label="SWA")
            plt.plot(epochs, cwa, label="CWA")
            plt.plot(epochs, hwa, label="HWA")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"Dataset: SPR_BENCH | Val Metrics (embed_dim={emb})")
            plt.legend()
            fname = f"SPR_BENCH_metrics_embed{emb}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating metric plot for emb={emb}: {e}")
            plt.close()

    # -------- summary plot across embedding sizes --------
    try:
        test_hwa = [data_tree[emb]["test_metrics"]["HWA"] for emb in embed_dims]
        plt.figure()
        plt.plot([int(e) for e in embed_dims], test_hwa, marker="o")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Test HWA")
        plt.title("Dataset: SPR_BENCH | Test HWA vs Embedding Dimension")
        fname = "SPR_BENCH_test_HWA_vs_embedding.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()
