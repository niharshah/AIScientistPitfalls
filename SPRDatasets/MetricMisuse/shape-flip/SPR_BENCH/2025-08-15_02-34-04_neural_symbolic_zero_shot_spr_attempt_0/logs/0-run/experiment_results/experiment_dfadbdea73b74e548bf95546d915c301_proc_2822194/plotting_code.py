import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment results --------
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is None or "hybrid" not in exp:
    print("No valid experiment data found; exiting.")
else:
    run = exp["hybrid"]

    # handy arrays
    tr_loss = run["losses"]["train"]
    val_loss = run["losses"]["val"]
    tr_swa = run["metrics"]["train"]
    val_swa = run["metrics"]["val"]
    test_swa = run["metrics"]["test"]
    preds = run.get("predictions", [])
    gts = run.get("ground_truth", [])

    # ---------- 1) loss curves ----------
    try:
        plt.figure()
        x = np.arange(len(tr_loss))
        plt.plot(x, tr_loss, "--", label="train")
        plt.plot(x, val_loss, "-", label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Hybrid Model\nTrain vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "spr_hybrid_loss_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- 2) SWA curves ----------
    try:
        plt.figure()
        x = np.arange(len(tr_swa))
        plt.plot(x, tr_swa, "--", label="train")
        plt.plot(x, val_swa, "-", label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH Hybrid Model\nTrain vs Validation SWA")
        plt.legend()
        fname = os.path.join(working_dir, "spr_hybrid_swa_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # ---------- 3) final SWA bar chart ----------
    try:
        plt.figure()
        bars = ["train", "validation", "test"]
        vals = [
            tr_swa[-1] if tr_swa else 0.0,
            val_swa[-1] if val_swa else 0.0,
            test_swa if test_swa is not None else 0.0,
        ]
        plt.bar(bars, vals, color=["#72bcd4", "#3896c1", "#1f77b4"])
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("SPR_BENCH Hybrid Model\nFinal SWA Scores")
        fname = os.path.join(working_dir, "spr_hybrid_final_swa_bar.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating SWA bar chart: {e}")
        plt.close()

    # ---------- 4) confusion matrix ----------
    try:
        if preds and gts:
            labels = sorted(set(gts))
            mat = np.zeros((len(labels), len(labels)), dtype=int)
            lab2idx = {l: i for i, l in enumerate(labels)}
            for t, p in zip(gts, preds):
                mat[lab2idx[t], lab2idx[p]] += 1
            plt.figure()
            im = plt.imshow(mat, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(range(len(labels)), labels)
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, mat[i, j], ha="center", va="center", color="black")
            plt.title("SPR_BENCH Hybrid Model\nConfusion Matrix (Test Set)")
            fname = os.path.join(working_dir, "spr_hybrid_confusion_matrix.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved {fname}")
            plt.close()
        else:
            print("No predictions/ground_truth found; skipping confusion matrix.")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ----------- print key metric -----------
    print(f"Test Shape-Weighted Accuracy: {test_swa:.4f}")
