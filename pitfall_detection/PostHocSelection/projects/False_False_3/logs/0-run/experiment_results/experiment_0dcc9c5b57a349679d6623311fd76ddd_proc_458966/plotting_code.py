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


def safe_close():
    if plt.get_fignums():
        plt.close()


for dname, d in experiment_data.items():
    losses_tr, losses_val = d.get("losses", {}).get("train", []), d.get(
        "losses", {}
    ).get("val", [])
    val_metrics = d.get("metrics", {}).get("val", [])
    preds, gts = d.get("predictions", []), d.get("ground_truth", [])

    # ---------- Figure 1: loss curves ----------
    try:
        if losses_tr and losses_val:
            plt.figure()
            plt.plot(losses_tr, "--", label="train")
            plt.plot(losses_val, "-", label="val")
            plt.title(f"{dname} Training vs Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
        safe_close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        safe_close()

    # ---------- Figure 2: validation SWA ----------
    try:
        if val_metrics:
            plt.figure()
            plt.plot(val_metrics, marker="o")
            plt.title(f"{dname} Validation Shape-Weighted Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("SWA")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dname}_val_SWA.png"))
        safe_close()
    except Exception as e:
        print(f"Error creating SWA plot for {dname}: {e}")
        safe_close()

    # ---------- Figure 3: confusion matrix ----------
    try:
        if preds and gts:
            labels = sorted(set(gts))
            idx = {l: i for i, l in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                cm[idx[t], idx[p]] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.title(f"{dname} Confusion Matrix")
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
                        fontsize=5,
                    )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
        safe_close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        safe_close()

    # ---------- print summary metrics ----------
    try:
        final_swa = val_metrics[-1] if val_metrics else None
        acc = np.mean(np.array(preds) == np.array(gts)) if preds and gts else None
        print(
            f"{dname}: final_val_SWA={final_swa:.4f}"
            if final_swa is not None
            else f"{dname}: no val SWA"
        )
        if acc is not None:
            print(f"{dname}: test_accuracy={acc:.4f}")
    except Exception as e:
        print(f"Error printing metrics for {dname}: {e}")
