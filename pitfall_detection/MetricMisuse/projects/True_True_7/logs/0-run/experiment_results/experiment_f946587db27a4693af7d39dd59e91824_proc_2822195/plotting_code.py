import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# load data
try:
    exp = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = exp.get("SPR_BENCH", None)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    data = None

# ------------------------------------------------------------------ #
if data is not None:
    train_loss = np.array(data["losses"]["train"])
    val_loss = np.array(data["losses"]["val"])
    train_swa = np.array(data["metrics"]["train_swa"])
    val_swa = np.array(data["metrics"]["val_swa"])
    test_swa = data["metrics"]["test_swa"]
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])

    # 1) loss curves ------------------------------------------------ #
    try:
        plt.figure(figsize=(6, 4))
        x = np.arange(len(train_loss))
        plt.plot(x, train_loss, ls="--", label="train")
        plt.plot(x, val_loss, ls="-", label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nTrain (dashed) vs Validation (solid)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) SWA curves ------------------------------------------------- #
    try:
        plt.figure(figsize=(6, 4))
        x = np.arange(len(train_swa))
        plt.plot(x, train_swa, ls="--", label="train")
        plt.plot(x, val_swa, ls="-", label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Sequence-Weighted Accuracy (SWA)")
        plt.title("SPR_BENCH SWA Curves\nTrain (dashed) vs Validation (solid)")
        plt.legend()
        fname = os.path.join(working_dir, "spr_bench_swa_curves.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating SWA plot: {e}")
        plt.close()

    # 3) test SWA bar ---------------------------------------------- #
    try:
        plt.figure(figsize=(4, 4))
        plt.bar(["test"], [test_swa], color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("SWA")
        plt.title("SPR_BENCH Final Test SWA")
        fname = os.path.join(working_dir, "spr_bench_test_swa_bar.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating test SWA bar: {e}")
        plt.close()

    # 4) confusion matrix ------------------------------------------ #
    try:
        labels = sorted(np.unique(np.concatenate([gts, preds])))
        lbl2idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for g, p in zip(gts, preds):
            cm[lbl2idx[g], lbl2idx[p]] += 1
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(np.arange(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # print metric -------------------------------------------------- #
    print(f"Test SWA = {test_swa:.4f}")
else:
    print("No experiment data found; no plots generated.")
