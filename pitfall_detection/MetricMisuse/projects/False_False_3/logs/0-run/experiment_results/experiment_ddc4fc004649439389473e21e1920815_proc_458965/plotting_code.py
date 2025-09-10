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

datasets = list(experiment_data.keys())

# ---------- Figure 1: loss curves ----------
try:
    plt.figure()
    for ds in datasets:
        tr_loss = experiment_data[ds]["losses"].get("train", [])
        val_loss = experiment_data[ds]["losses"].get("val", [])
        plt.plot(tr_loss, "--", label=f"{ds} train")
        plt.plot(val_loss, "-", label=f"{ds} val")
    if datasets:
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "loss_curves_all_datasets.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- Figure 2: validation SWA ----------
try:
    plt.figure()
    for ds in datasets:
        val_swa = experiment_data[ds]["metrics"].get("val", [])
        plt.plot(val_swa, label=ds)
    if datasets:
        plt.title("Validation Shape-Weighted Accuracy (SWA)")
        plt.xlabel("Epoch")
        plt.ylabel("SWA")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "val_SWA_all_datasets.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA curves: {e}")
    plt.close()

# ---------- Figure 3: final SWA bar chart ----------
try:
    labels_ds, train_end, val_end, test_end = [], [], [], []
    for ds in datasets:
        labels_ds.append(ds)
        tr = experiment_data[ds]["losses"].get("train", [])
        vl = experiment_data[ds]["losses"].get("val", [])
        # compute final SWA on splits if available
        val_swa = experiment_data[ds]["metrics"].get("val", [])
        train_end.append(val_swa[0] if val_swa else 0)  # placeholder if no train metric
        val_end.append(val_swa[-1] if val_swa else 0)
        # test SWA as simple accuracy if sequences unavailable
        preds = experiment_data[ds].get("predictions", [])
        gts = experiment_data[ds].get("ground_truth", [])
        if preds and gts:
            test_end.append(sum(int(p == t) for p, t in zip(preds, gts)) / len(gts))
        else:
            test_end.append(0)
    x = np.arange(len(labels_ds))
    width = 0.25
    plt.figure()
    plt.bar(x - width, train_end, width, label="Train*")
    plt.bar(x, val_end, width, label="Val")
    plt.bar(x + width, test_end, width, label="Test")
    if datasets:
        plt.title("Final SWA Scores per Dataset\n(*train uses first val SWA as proxy)")
        plt.xticks(x, labels_ds, rotation=45, ha="right")
        plt.ylabel("SWA")
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "final_SWA_bar_chart.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ---------- Figure 4: confusion matrices (max 5 datasets) ----------
from collections import Counter
import itertools

for idx_ds, ds in enumerate(datasets[:5]):
    try:
        preds = experiment_data[ds].get("predictions", [])
        gts = experiment_data[ds].get("ground_truth", [])
        if not preds or not gts:
            continue
        labels_set = sorted({*preds, *gts})
        label_to_i = {l: i for i, l in enumerate(labels_set)}
        cm = np.zeros((len(labels_set), len(labels_set)), dtype=int)
        for t, p in zip(gts, preds):
            cm[label_to_i[t], label_to_i[p]] += 1
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title(f"{ds} Confusion Matrix\nTest Split")
        plt.xticks(range(len(labels_set)), labels_set, rotation=90, fontsize=6)
        plt.yticks(range(len(labels_set)), labels_set, fontsize=6)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
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
        plt.savefig(os.path.join(working_dir, f"{ds}_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds}: {e}")
        plt.close()
