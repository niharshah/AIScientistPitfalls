import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------------------------------------------------------------------
# helper to subsample long epoch lists
def _subsample(xs, ys, max_pts=100):
    if len(xs) <= max_pts:
        return xs, ys
    step = int(np.ceil(len(xs) / max_pts))
    return xs[::step], ys[::step]


# ---------------------------------------------------------------------
# per-dataset plots
for dset_name, d in list(experiment_data.items())[:5]:  # ≤5 datasets
    # -------- Learning curves: BWA
    try:
        epochs = np.arange(1, len(d["metrics"]["train"]) + 1)
        train_bwa = np.array(d["metrics"]["train"])
        val_bwa = np.array(d["metrics"]["val"])
        epochs_s, train_bwa_s = _subsample(epochs, train_bwa)
        _, val_bwa_s = _subsample(epochs, val_bwa)
        plt.figure()
        plt.plot(epochs_s, train_bwa_s, label="Train BWA")
        plt.plot(epochs_s, val_bwa_s, label="Validation BWA")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title(f"{dset_name} – BWA Learning Curve")
        plt.legend()
        plt.tight_layout()
        fname = f"{dset_name.lower()}_bwa_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error BWA curve ({dset_name}): {e}")
        plt.close()

    # -------- Learning curves: Loss
    try:
        epochs = np.arange(1, len(d["losses"]["train"]) + 1)
        train_loss = np.array(d["losses"]["train"])
        val_loss = np.array(d["losses"]["val"])
        epochs_s, train_loss_s = _subsample(epochs, train_loss)
        _, val_loss_s = _subsample(epochs, val_loss)
        plt.figure()
        plt.plot(epochs_s, train_loss_s, label="Train Loss")
        plt.plot(epochs_s, val_loss_s, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dset_name} – Loss Curve")
        plt.legend()
        plt.tight_layout()
        fname = f"{dset_name.lower()}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error Loss curve ({dset_name}): {e}")
        plt.close()

    # -------- Learning curves: StrWA
    try:
        epochs = np.arange(1, len(d["StrWA"]["train"]) + 1)
        train_s = np.array(d["StrWA"]["train"])
        val_s = np.array(d["StrWA"]["val"])
        epochs_s, train_s_s = _subsample(epochs, train_s)
        _, val_s_s = _subsample(epochs, val_s)
        plt.figure()
        plt.plot(epochs_s, train_s_s, label="Train StrWA")
        plt.plot(epochs_s, val_s_s, label="Validation StrWA")
        plt.xlabel("Epoch")
        plt.ylabel("StrWA")
        plt.title(f"{dset_name} – Structural WA Curve")
        plt.legend()
        plt.tight_layout()
        fname = f"{dset_name.lower()}_strwa_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error StrWA curve ({dset_name}): {e}")
        plt.close()

    # -------- Confusion Matrix
    try:
        preds = np.array(d.get("predictions", []))
        gts = np.array(d.get("ground_truth", []))
        if preds.size and gts.size:
            num_cls = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for gt, pr in zip(gts, preds):
                cm[gt, pr] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dset_name} – Confusion Matrix")
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center", fontsize=7)
            plt.tight_layout()
            fname = f"{dset_name.lower()}_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error Confusion Matrix ({dset_name}): {e}")
        plt.close()

# ---------------------------------------------------------------------
# cross-dataset test BWA comparison
try:
    names, test_bwa = [], []
    for dn, dd in experiment_data.items():
        if "metrics" in dd and dd["metrics"]["val"]:
            names.append(dn)
            test_bwa.append(dd["metrics"]["val"][-1])
    if names:
        plt.figure()
        x = np.arange(len(names))
        plt.bar(x, test_bwa, color="skyblue")
        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylabel("Final Validation BWA")
        plt.title("Dataset Comparison: Final Validation BWA")
        plt.tight_layout()
        fname = "dataset_bwa_comparison.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error dataset comparison bar chart: {e}")
    plt.close()
