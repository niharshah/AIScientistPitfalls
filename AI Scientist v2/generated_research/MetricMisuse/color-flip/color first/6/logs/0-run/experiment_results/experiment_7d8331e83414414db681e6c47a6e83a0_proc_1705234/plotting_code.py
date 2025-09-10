import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data = experiment_data.get("final_state_pool", {}).get("spr", {})

losses_tr = data.get("losses", {}).get("train", [])
losses_val = data.get("losses", {}).get("val", [])
metrics_val = data.get("metrics", {}).get("val", [])
preds = np.array(data.get("predictions", []))
golds = np.array(data.get("ground_truth", []))

saved_paths = []

# -------------------------------------------------------------------------
# 1) Loss curves
try:
    if losses_tr and losses_val:
        ep_tr, loss_tr = zip(*losses_tr)
        ep_val, loss_val = zip(*losses_val)
        plt.figure()
        plt.plot(ep_tr, loss_tr, label="Train")
        plt.plot(ep_val, loss_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        path = os.path.join(working_dir, "spr_loss_curves.png")
        plt.savefig(path)
        saved_paths.append(path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 2) Validation metrics curves
try:
    if metrics_val:
        ep, cwa, swa, pcwa = [], [], [], []
        for e, d in metrics_val:
            ep.append(e)
            cwa.append(d["CWA"])
            swa.append(d["SWA"])
            pcwa.append(d["PCWA"])
        plt.figure()
        plt.plot(ep, cwa, label="CWA")
        plt.plot(ep, swa, label="SWA")
        plt.plot(ep, pcwa, label="PCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR Validation Metrics")
        plt.legend()
        path = os.path.join(working_dir, "spr_validation_metrics.png")
        plt.savefig(path)
        saved_paths.append(path)
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 3) Confusion matrix
try:
    if preds.size and golds.size:
        labels = sorted(list(set(golds) | set(preds)))
        lab2idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(golds, preds):
            cm[lab2idx[t], lab2idx[p]] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(np.arange(len(labels)), labels, rotation=90)
        plt.yticks(np.arange(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR Confusion Matrix")
        path = os.path.join(working_dir, "spr_confusion_matrix.png")
        plt.savefig(path, bbox_inches="tight")
        saved_paths.append(path)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print("Saved figures:", saved_paths)
