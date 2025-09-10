import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    datasets = list(experiment_data.keys())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    datasets = []

# ---------------------------------------------------------------------
# 1) Train/Validation curves (loss & BWA) – at most 5 datasets
for i, dname in enumerate(datasets[:5]):
    d = experiment_data[dname]
    epochs = np.arange(1, len(d["losses"]["train"]) + 1)

    # ---- Loss curve
    try:
        plt.figure()
        plt.plot(epochs, d["losses"]["train"], label="Train Loss")
        plt.plot(epochs, d["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname.upper()} Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        fname = f"{dname}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error plotting loss curve for {dname}: {e}")
        plt.close()

    # ---- BWA curve
    try:
        train_bwa = [m["BWA"] for m in d["metrics"]["train"]]
        val_bwa = [m["BWA"] for m in d["metrics"]["val"]]
        plt.figure()
        plt.plot(epochs, train_bwa, label="Train BWA")
        plt.plot(epochs, val_bwa, label="Validation BWA")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title(f"{dname.upper()} BWA Learning Curve")
        plt.legend()
        plt.tight_layout()
        fname = f"{dname}_bwa_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error plotting BWA curve for {dname}: {e}")
        plt.close()

# ---------------------------------------------------------------------
# 2) Test-set BWA comparison across datasets
try:
    names, bwa_vals = [], []
    for dname, d in experiment_data.items():
        if "test_metrics" in d and "BWA" in d["test_metrics"]:
            names.append(dname)
            bwa_vals.append(d["test_metrics"]["BWA"])
    if names:
        plt.figure()
        xpos = np.arange(len(names))
        plt.bar(xpos, bwa_vals, color="steelblue")
        plt.xticks(xpos, names, rotation=45, ha="right")
        plt.ylabel("Test BWA")
        plt.title("Test-set BWA Comparison Across Datasets")
        plt.tight_layout()
        fname = "datasets_test_bwa_comparison.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating cross-dataset BWA bar chart: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 3) Confusion matrix for each dataset (<=5)
for i, dname in enumerate(datasets[:5]):
    d = experiment_data[dname]
    if len(d.get("predictions", [])) == 0:
        continue
    try:
        preds = np.array(d["predictions"])
        gts = np.array(d["ground_truth"])
        num_classes = int(max(preds.max(), gts.max()) + 1)
        conf = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts, preds):
            conf[gt, pr] += 1
        plt.figure(figsize=(6, 5))
        im = plt.imshow(conf, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{dname.upper()} Confusion Matrix")
        for (r, c), v in np.ndenumerate(conf):
            plt.text(c, r, str(v), ha="center", va="center", fontsize=8)
        plt.tight_layout()
        fname = f"{dname}_confusion_matrix.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error plotting confusion matrix for {dname}: {e}")
        plt.close()
