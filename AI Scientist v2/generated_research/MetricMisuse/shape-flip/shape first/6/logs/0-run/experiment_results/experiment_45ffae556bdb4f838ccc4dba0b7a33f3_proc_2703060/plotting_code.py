import matplotlib.pyplot as plt
import numpy as np
import os

# -------- setup --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load artefacts --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    rec = experiment_data["HybridTransformer"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data, rec = None, None

if rec:
    epochs = range(1, len(rec["losses"]["train"]) + 1)

    # 1) Loss curves ----------------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, rec["losses"]["train"], label="Train Loss")
        plt.plot(epochs, rec["losses"]["val"], label="Val Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Val")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) Validation SWA -------------------------------------------------------
    try:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, rec["SWA"]["val"], color="green")
        plt.title("SPR_BENCH Validation Shape-Weighted Accuracy (SWA)")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        fname = os.path.join(working_dir, "SPR_BENCH_val_SWA.png")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 3) Confusion matrix -----------------------------------------------------
    try:
        preds, gts = np.array(rec["predictions"]), np.array(rec["ground_truth"])
        num_labels = max(preds.max(), gts.max()) + 1 if preds.size else 0
        if num_labels and preds.size:
            cm = np.zeros((num_labels, num_labels), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure(figsize=(5, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.title("SPR_BENCH Test Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        else:
            print("Confusion matrix skipped: no prediction data")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # 4) Val-vs-Test SWA bar --------------------------------------------------
    try:
        best_val_swa = max(rec["SWA"]["val"]) if rec["SWA"]["val"] else None
        test_swa = rec["SWA"]["test"]
        if best_val_swa is not None:
            plt.figure(figsize=(4, 4))
            plt.bar(
                ["Best Val", "Test"],
                [best_val_swa, test_swa],
                color=["orange", "skyblue"],
            )
            plt.ylim(0, 1)
            plt.title("SPR_BENCH: Best Val vs Test SWA")
            fname = os.path.join(working_dir, "SPR_BENCH_val_vs_test_SWA.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating val vs test bar: {e}")
        plt.close()

    # -------- print metrics -------------
    test_metrics = rec["metrics"]["test"]
    print(
        f"Test Loss: {test_metrics['loss']:.4f} | Test SWA: {test_metrics['SWA']:.3f}"
    )
