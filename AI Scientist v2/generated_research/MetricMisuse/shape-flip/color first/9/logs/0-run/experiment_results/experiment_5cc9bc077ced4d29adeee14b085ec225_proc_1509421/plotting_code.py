import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------------- helper --------------------
def sdwa_metric(seqs, y_true, y_pred):
    def _uniq_shapes(seq):
        return len(set(tok[0] for tok in seq.split()))

    def _uniq_colors(seq):
        return len(set(tok[1] for tok in seq.split()))

    weights = [_uniq_shapes(s) + _uniq_colors(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1)


# ---------------- plotting ------------------
plot_count = 0
MAX_PLOTS = 5

for dname, d in experiment_data.items():
    epochs = d.get("epochs", [])
    tr_loss = d.get("losses", {}).get("train", [])
    val_loss = d.get("losses", {}).get("val", [])
    tr_met = d.get("metrics", {}).get("train", [])
    val_met = d.get("metrics", {}).get("val", [])
    preds = d.get("predictions", [])
    gts = d.get("ground_truth", [])
    seqs = d.get("seqs", []) if "seqs" in d else []  # may not exist

    # 1) Loss curve
    if plot_count < MAX_PLOTS:
        try:
            plt.figure()
            plt.plot(epochs, tr_loss, label="train")
            plt.plot(epochs, val_loss, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} – Loss Curves")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname.lower()}_loss_curve.png")
            plt.savefig(fname, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {dname}: {e}")
            plt.close()
        plot_count += 1

    # 2) Metric curve
    if plot_count < MAX_PLOTS and tr_met and val_met:
        try:
            plt.figure()
            plt.plot(epochs, tr_met, label="train")
            plt.plot(epochs, val_met, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("SDWA")
            plt.title(f"{dname} – SDWA Curves")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname.lower()}_sdwa_curve.png")
            plt.savefig(fname, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating metric curve for {dname}: {e}")
            plt.close()
        plot_count += 1

    # 3) Confusion matrix
    if plot_count < MAX_PLOTS and preds and gts:
        try:
            n_classes = max(max(preds), max(gts)) + 1
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(n_classes):
                for j in range(n_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dname} – Confusion Matrix")
            fname = os.path.join(working_dir, f"{dname.lower()}_confusion_matrix.png")
            plt.savefig(fname, dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dname}: {e}")
            plt.close()
        plot_count += 1

    # ---- print evaluation metric ----
    if preds and gts and seqs:
        sdwa = sdwa_metric(seqs, gts, preds)
        print(f"{dname} Test SDWA: {sdwa:.4f}")
    elif preds and gts:
        acc = np.mean(np.array(preds) == np.array(gts))
        print(f"{dname} Test Accuracy: {acc:.4f}")
