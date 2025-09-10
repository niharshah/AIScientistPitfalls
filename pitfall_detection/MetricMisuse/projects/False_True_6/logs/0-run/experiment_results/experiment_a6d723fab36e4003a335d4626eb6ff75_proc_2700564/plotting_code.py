import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load experiment artefacts ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    runs = experiment_data["embedding_dim"]["SPR_BENCH"]["runs"]
    # ---------- 1) summary test accuracy vs emb dim ----------
    try:
        dims, test_accs = zip(
            *[(d, r["metrics"]["test"]["acc"]) for d, r in runs.items()]
        )
        plt.figure(figsize=(6, 4))
        plt.bar([str(d) for d in dims], test_accs, color="skyblue")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Test Accuracy vs Embedding Dimension")
        plt.xlabel("Embedding Dimension"), plt.ylabel("Accuracy")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_acc_vs_emb_dim.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

    # ---------- 2-5) per-dim learning curves ----------
    for i, (emb_dim, record) in enumerate(
        sorted(runs.items())[:5]
    ):  # safeguard for many dims
        try:
            epochs = range(1, len(record["losses"]["train"]) + 1)
            train_losses = record["losses"]["train"]
            val_losses = record["losses"]["val"]
            val_accs = [m["acc"] for m in record["metrics"]["val"]]

            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
            ax1.plot(epochs, val_losses, "r-o", label="Val Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend(loc="upper left")

            ax2 = ax1.twinx()
            ax2.plot(epochs, val_accs, "g-s", label="Val Acc")
            ax2.set_ylabel("Accuracy")
            ax2.set_ylim(0, 1)
            ax2.legend(loc="upper right")

            plt.title(
                f"SPR_BENCH Learning Curves (Embedding={emb_dim})\nLeft: Loss, Right: Accuracy"
            )
            plt.tight_layout()
            fname = os.path.join(
                working_dir, f"SPR_BENCH_learning_curves_emb{emb_dim}.png"
            )
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error creating plot for emb_dim={emb_dim}: {e}")
            plt.close()
