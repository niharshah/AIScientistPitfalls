import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

# -------- Plot 1: loss curves -------------------------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    plt.figure()
    for dset, d in experiment_data.items():
        if d["losses"]["train"]:
            e, l = zip(*d["losses"]["train"])
            plt.plot(e, l, "--", label=f"{dset}-train")
        if d["losses"]["val"]:
            e, l = zip(*d["losses"]["val"])
            plt.plot(e, l, "-", label=f"{dset}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    fname = os.path.join(
        working_dir, "_".join(experiment_data.keys()) + "_loss_curves.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# -------- Plot 2: validation CSHM --------------------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    plt.figure()
    for dset, d in experiment_data.items():
        val = d["metrics"]["val"]
        if val:
            epochs = [t[0] for t in val]
            cshm = [t[3] for t in val]  # (epoch, cwa, swa, cshm, ocga)
            plt.plot(epochs, cshm, label=f"{dset}-CSHM")
    plt.xlabel("Epoch")
    plt.ylabel("CSHM")
    plt.title("Validation Colour–Shape Harmonic Mean")
    plt.legend()
    fname = os.path.join(
        working_dir, "_".join(experiment_data.keys()) + "_val_cshm.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CSHM plot: {e}")
    plt.close()

# -------- Plot 3: confusion matrix (one per dataset, max 5) -------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    for idx, (dset, d) in enumerate(experiment_data.items()):
        if idx >= 5:  # safety cap
            break
        preds, gts = np.array(d["predictions"]), np.array(d["ground_truth"])
        if len(preds) == 0:
            continue
        n_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"{dset.upper()} Test Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(n_cls))
        plt.yticks(range(n_cls))
        fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# -------- Plot 4: final test accuracy bar chart ------------------------------
try:
    if experiment_data is None:
        raise ValueError("No experiment data loaded.")
    names, accs = [], []
    for dset, d in experiment_data.items():
        preds, gts = np.array(d["predictions"]), np.array(d["ground_truth"])
        if len(preds):
            names.append(dset)
            accs.append((preds == gts).mean())
    if accs:
        plt.figure()
        plt.bar(names, accs, color="orange")
        for i, v in enumerate(accs):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
        plt.ylim(0, 1.05)
        plt.ylabel("Accuracy")
        plt.title("Final Test Accuracy per Dataset")
        fname = os.path.join(working_dir, "test_accuracy_summary.png")
        plt.savefig(fname)
        plt.close()
        print("Test accuracies:", dict(zip(names, accs)))
except Exception as e:
    print(f"Error creating accuracy bar chart: {e}")
    plt.close()
