import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = exp["emb_dim"]["SPR_BENCH"]["runs"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = []

if runs:  # proceed only if we actually have data
    # collect data
    emb_dims = [r["emb_dim"] for r in runs]
    train_loss_curves = [r["losses"]["train"] for r in runs]
    val_loss_curves = [r["losses"]["val"] for r in runs]
    val_hwa_curves = [[m["hwa"] for m in r["metrics"]["val"]] for r in runs]
    test_hwa = [r["metrics"]["test"]["hwa"] for r in runs]

    # 1) combined loss curves
    try:
        plt.figure()
        for emb, tr, vl in zip(emb_dims, train_loss_curves, val_loss_curves):
            ep = range(1, len(tr) + 1)
            plt.plot(ep, tr, label=f"train (emb={emb})", linestyle="-")
            plt.plot(ep, vl, label=f"val (emb={emb})", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(working_dir, "SPR_BENCH_embSweep_loss_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve figure: {e}")
        plt.close()

    # 2) combined HWA curves
    try:
        plt.figure()
        for emb, hwa in zip(emb_dims, val_hwa_curves):
            ep = range(1, len(hwa) + 1)
            plt.plot(ep, hwa, label=f"emb={emb}")
        plt.xlabel("Epoch")
        plt.ylabel("Harmonic Weighted Accuracy")
        plt.title("SPR_BENCH: Validation HWA Across Epochs")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(working_dir, "SPR_BENCH_embSweep_HWA_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA curve figure: {e}")
        plt.close()

    # 3) bar chart of final test HWA
    try:
        plt.figure()
        plt.bar([str(e) for e in emb_dims], test_hwa, color="skyblue")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Test HWA")
        plt.title(
            "SPR_BENCH: Final Test HWA by Embedding Size\n(Left: smaller emb, Right: larger emb)"
        )
        plt.tight_layout()
        save_path = os.path.join(working_dir, "SPR_BENCH_test_HWA_bar.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating test HWA bar figure: {e}")
        plt.close()

    # ---------- print summary ----------
    best_idx = int(np.argmax(test_hwa))
    print("Test HWA per embedding:", dict(zip(emb_dims, test_hwa)))
    print(f"Best embedding dim: {emb_dims[best_idx]} with HWA={test_hwa[best_idx]:.3f}")
else:
    print("No runs were found inside experiment_data.npy.")
