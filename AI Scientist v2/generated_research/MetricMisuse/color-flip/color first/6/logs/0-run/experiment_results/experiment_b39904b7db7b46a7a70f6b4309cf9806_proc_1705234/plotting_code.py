import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------------------------
# paths / load data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ed = experiment_data.get("unmasked_mean_pooling", {}).get("dual_channel", {})

# -------------------------------------------------------------------------
# 1) loss curves
try:
    tl = ed["losses"]["train"]  # list[(epoch, loss)]
    vl = ed["losses"]["val"]
    ep_t, tr = zip(*tl)
    ep_v, va = zip(*vl)

    plt.figure()
    plt.plot(ep_t, tr, label="Train Loss")
    plt.plot(ep_v, va, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Dual-Channel / Unmasked-Mean – Train vs Val Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "dual_channel_unmasked_mean_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 2) metric curves (CWA, SWA, PCWA)
try:
    m_val = ed["metrics"]["val"]  # list[(epoch, dict)]
    ep, cwa, swa, pcwa = [], [], [], []
    for e, d in m_val:
        ep.append(e)
        cwa.append(d["CWA"])
        swa.append(d["SWA"])
        pcwa.append(d["PCWA"])

    plt.figure()
    plt.plot(ep, cwa, label="CWA")
    plt.plot(ep, swa, label="SWA")
    plt.plot(ep, pcwa, label="PCWA")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Accuracy")
    plt.title("Dual-Channel / Unmasked-Mean – Val Metrics over Epochs")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "dual_channel_unmasked_mean_metric_curves.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating metric curves: {e}")
    plt.close()

# -------------------------------------------------------------------------
# 3) confusion matrix on test data
try:
    y_true = ed["ground_truth"]
    y_pred = ed["predictions"]
    labels = sorted(set(y_true) | set(y_pred))
    lbl2idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[lbl2idx[t], lbl2idx[p]] += 1

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Dual-Channel / Unmasked-Mean – Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "dual_channel_unmasked_mean_confusion_matrix.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
