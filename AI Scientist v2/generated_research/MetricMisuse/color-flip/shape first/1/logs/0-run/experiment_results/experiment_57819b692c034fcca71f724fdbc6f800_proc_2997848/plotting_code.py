import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR_BENCH", {})


def _style(idx):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[idx % len(colors)], "-" if idx < len(colors) else "--"


# 1. Pre-training loss curve
try:
    losses_pt = spr.get("losses", {}).get("pretrain", [])
    if losses_pt:
        plt.figure()
        plt.plot(range(1, len(losses_pt) + 1), losses_pt, label="pretrain")
        plt.title("SPR_BENCH: Pre-training Loss vs Epochs")
        plt.xlabel("Pre-training Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_pretrain_loss.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating pre-training loss plot: {e}")
    plt.close()

# 2. Fine-tuning train/val loss
try:
    tr = spr.get("losses", {}).get("train", [])
    vl = spr.get("losses", {}).get("val", [])
    if tr or vl:
        plt.figure()
        if tr:
            plt.plot(range(1, len(tr) + 1), tr, label="train", linestyle="-")
        if vl:
            plt.plot(range(1, len(vl) + 1), vl, label="val", linestyle="--")
        plt.title("SPR_BENCH: Fine-tuning Loss")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel("Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_finetune_loss.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating fine-tuning loss plot: {e}")
    plt.close()


# Helper to plot metric curves
def plot_metric(metric_key, pretty_name):
    try:
        vals = [
            d.get(metric_key)
            for d in spr.get("metrics", {}).get("val", [])
            if metric_key in d
        ]
        if vals:
            plt.figure()
            plt.plot(range(1, len(vals) + 1), vals, label=pretty_name)
            plt.title(f"SPR_BENCH: {pretty_name} vs Fine-tuning Epochs")
            plt.xlabel("Fine-tuning Epoch")
            plt.ylabel(pretty_name)
            plt.legend()
            fname = os.path.join(working_dir, f"SPR_BENCH_{metric_key}_curve.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
    except Exception as e:
        print(f"Error creating {pretty_name} plot: {e}")
        plt.close()


# 3–5. Metric curves
plot_metric("SWA", "Shape-Weighted Accuracy")
plot_metric("CWA", "Color-Weighted Accuracy")
plot_metric("CompWA", "Complexity-Weighted Accuracy")

# 6. Confusion matrix (optional fifth figure, only if data present)
try:
    preds = np.array(spr.get("predictions", []))
    gts = np.array(spr.get("ground_truth", []))
    if preds.size and gts.size and preds.shape == gts.shape:
        classes = np.unique(np.concatenate([preds, gts]))
        cm = np.zeros((classes.size, classes.size), dtype=int)
        for p, t in zip(preds, gts):
            i = np.where(classes == t)[0][0]
            j = np.where(classes == p)[0][0]
            cm[i, j] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix")
        plt.xticks(range(classes.size), classes, rotation=45)
        plt.yticks(range(classes.size), classes)
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
