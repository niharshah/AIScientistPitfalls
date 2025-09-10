import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load data ----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR_BENCH", {})
losses = spr.get("losses", {})
metrics_val = spr.get("metrics", {}).get("val", [])
preds = spr.get("predictions", [])
gts = spr.get("ground_truth", [])


def split_xy(tuples):
    if not tuples:
        return np.array([]), np.array([])
    arr = np.array(tuples, dtype=float)
    return arr[:, 0], arr[:, 1]


# ---- plot 1: loss curves ----
try:
    plt.figure()
    e_tr, l_tr = split_xy(losses.get("train", []))
    e_val, l_val = split_xy(losses.get("val", []))
    e_con, l_con = split_xy(losses.get("contrastive", []))
    if e_tr.size:
        plt.plot(e_tr, l_tr, label="Train Loss")
    if e_val.size:
        plt.plot(e_val, l_val, label="Val Loss")
    if e_con.size:
        plt.plot(e_con, l_con, label="Contrastive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("SPR_BENCH: Training & Validation Losses")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---- plot 2: validation metrics ----
try:
    plt.figure()
    if metrics_val:
        mv = np.array(metrics_val, dtype=float)
        epochs, swa, cwa, hwa = mv[:, 0], mv[:, 1], mv[:, 2], mv[:, 3]
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, hwa, label="HWA")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.title("SPR_BENCH: Validation Metrics Over Epochs")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_validation_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()

# ---- plot 3: confusion matrix ----
if preds and gts:
    try:
        import itertools

        classes = sorted(set(gts))
        idx = {c: i for i, c in enumerate(classes)}
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            cm[idx[t], idx[p]] += 1
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        for i, j in itertools.product(range(len(classes)), repeat=2):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

# ---- print a simple evaluation metric ----
if preds and gts:
    acc = np.mean(np.array(preds) == np.array(gts))
    print(f"Test Accuracy: {acc:.4f}")
