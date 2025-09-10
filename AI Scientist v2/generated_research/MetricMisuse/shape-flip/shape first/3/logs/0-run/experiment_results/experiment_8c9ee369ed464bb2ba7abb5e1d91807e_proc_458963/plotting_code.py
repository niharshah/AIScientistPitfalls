import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- paths & loading -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ------------- util for SWA (copied from training script) -------------
def count_shape_variety(seq: str) -> int:
    return len({tok[0] for tok in seq.split() if tok})


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(v if t == p else 0 for v, t, p in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ----------------- plotting -----------------
for ds_name, ds_dict in experiment_data.items():
    losses = ds_dict.get("losses", {})
    val_metrics = ds_dict.get("metrics", {}).get("val", [])
    preds = ds_dict.get("predictions", [])
    gts = ds_dict.get("ground_truth", [])

    # ---------- Plot 1: loss curves ----------
    try:
        if losses:
            plt.figure()
            if losses.get("train"):
                # subsample to at most 5 points
                x = np.linspace(
                    0, len(losses["train"]) - 1, min(5, len(losses["train"]))
                ).astype(int)
                plt.plot(x, np.array(losses["train"])[x], "--o", label="train")
            if losses.get("val"):
                x = np.linspace(
                    0, len(losses["val"]) - 1, min(5, len(losses["val"]))
                ).astype(int)
                plt.plot(x, np.array(losses["val"])[x], "-o", label="val")
            plt.title(f"{ds_name} Loss Curves")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # ---------- Plot 2: validation SWA ----------
    try:
        if val_metrics:
            plt.figure()
            x = np.linspace(0, len(val_metrics) - 1, min(5, len(val_metrics))).astype(
                int
            )
            plt.plot(x, np.array(val_metrics)[x], "-o")
            plt.title(f"{ds_name} Validation Shape-Weighted-Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_val_SWA.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {ds_name}: {e}")
        plt.close()

    # ---------- Plot 3: confusion matrix ----------
    try:
        if preds and gts:
            labels = sorted(set(gts))
            idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                if t in idx and p in idx:
                    cm[idx[t], idx[p]] += 1
            plt.figure(figsize=(6, 6))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.title(
                f"{ds_name} Confusion Matrix\nRows: Ground Truth, Cols: Predicted"
            )
            plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
            plt.yticks(range(len(labels)), labels, fontsize=6)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=6,
                    )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

    # ---------- Print final test SWA ----------
    try:
        seqs = ds_dict.get("test_sequences", [])  # may not exist
        if preds and gts and seqs:
            swa = shape_weighted_accuracy(seqs, gts, preds)
            print(f"{ds_name} final test SWA: {swa:.4f}")
    except Exception as e:
        print(f"Error computing final SWA for {ds_name}: {e}")
