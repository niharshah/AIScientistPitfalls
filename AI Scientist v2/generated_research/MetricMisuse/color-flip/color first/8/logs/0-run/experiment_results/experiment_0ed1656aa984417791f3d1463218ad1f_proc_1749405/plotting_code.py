import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("MeanPoolOnly", {}).get("SPR", {})
loss_tr = np.array(spr.get("losses", {}).get("train", []))  # (epoch, loss)
loss_val = np.array(spr.get("losses", {}).get("val", []))  # (epoch, loss)
metrics_val = np.array(
    spr.get("metrics", {}).get("val", [])
)  # (epoch, cwa, swa, hm, ocga)
preds = np.array(spr.get("predictions", []))
gts = np.array(spr.get("ground_truth", []))

# ---------- plot 1: loss curves ----------
try:
    if len(loss_tr) and len(loss_val):
        plt.figure()
        plt.plot(loss_tr[:, 0], loss_tr[:, 1], label="Train")
        plt.plot(loss_val[:, 0], loss_val[:, 1], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR Dataset – Loss Curves (Train vs Validation)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_loss_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------- plot 2: metric curves ----------
try:
    if len(metrics_val):
        plt.figure()
        ep, cwa, swa, hm, ocga = metrics_val.T
        plt.plot(ep, cwa, label="CWA")
        plt.plot(ep, swa, label="SWA")
        plt.plot(ep, hm, label="HM")
        plt.plot(ep, ocga, label="OCGA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR Dataset – Metric Curves (CWA, SWA, HM, OCGA)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_metric_curves.png"))
        plt.close()
except Exception as e:
    print(f"Error creating metric curve plot: {e}")
    plt.close()

# ---------- plot 3: confusion matrix ----------
try:
    if preds.size and gts.size:
        classes = np.unique(np.concatenate([preds, gts]))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(gts, preds):
            cm[np.where(classes == t)[0][0], np.where(classes == p)[0][0]] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR Dataset – Confusion Matrix (Ground Truth vs Predictions)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------- print evaluation metrics ----------
if preds.size and gts.size:
    accuracy = (preds == gts).mean()
    print(f"Test Accuracy: {accuracy:.3f}")

if len(metrics_val):
    last_ep, cwa, swa, hm, ocga = metrics_val[-1]
    print(
        f"Last Val Metrics (Epoch {int(last_ep)}): CWA={cwa:.3f}, SWA={swa:.3f}, HM={hm:.3f}, OCGA={ocga:.3f}"
    )
