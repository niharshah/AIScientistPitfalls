import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load experiment data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

mdl = "bi_lstm_backbone"
dset = "SPR_BENCH"
run = experiment_data.get(mdl, {}).get(dset, {})

loss_train = run.get("losses", {}).get("train", [])
loss_val = run.get("losses", {}).get("val", [])
metrics_val = run.get("metrics", {}).get("val", [])
preds = np.array(run.get("predictions", []))
gts = np.array(run.get("ground_truth", []))
wts = np.array(run.get("weights", []))
epochs = np.arange(1, len(loss_train) + 1)

# 1) Train vs Val loss
try:
    plt.figure()
    plt.plot(epochs, loss_train, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy Loss")
    plt.title(f"{mdl} on {dset} – Train vs Val Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{dset.lower()}_{mdl}_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Macro-F1
try:
    plt.figure()
    f1_vals = [m["macro_f1"] for m in metrics_val]
    plt.plot(epochs, f1_vals, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(f"{mdl} on {dset} – Validation Macro-F1")
    fname = os.path.join(working_dir, f"{dset.lower()}_{mdl}_macro_f1.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating Macro-F1 plot: {e}")
    plt.close()

# 3) CWA
try:
    plt.figure()
    cwa_vals = [m["cwa"] for m in metrics_val]
    plt.plot(epochs, cwa_vals, marker="s", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Class-weighted Accuracy")
    plt.title(f"{mdl} on {dset} – Validation CWA")
    fname = os.path.join(working_dir, f"{dset.lower()}_{mdl}_cwa.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# 4) Confusion Matrix (final epoch)
try:
    from sklearn.metrics import confusion_matrix

    if preds.size and gts.size:
        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{mdl} on {dset} – Confusion Matrix (Final)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        fname = os.path.join(working_dir, f"{dset.lower()}_{mdl}_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# 5) Weight histogram vs correctness
try:
    if preds.size and wts.size:
        correct = preds == gts
        plt.figure()
        plt.hist(wts[correct], bins=20, alpha=0.7, label="Correct", color="blue")
        plt.hist(wts[~correct], bins=20, alpha=0.7, label="Incorrect", color="orange")
        plt.xlabel("Example Weight")
        plt.ylabel("Count")
        plt.title(f"{mdl} on {dset} – Weight Distribution by Correctness")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset.lower()}_{mdl}_weight_histogram.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating weight histogram: {e}")
    plt.close()
