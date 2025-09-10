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
    experiment_data = {}

for dset, ed in experiment_data.items():
    epochs = ed.get("epochs", [])
    tr_loss = ed.get("losses", {}).get("train", [])
    val_loss = ed.get("losses", {}).get("val", [])
    tr_pcwa = [m["pcwa"] for m in ed.get("metrics", {}).get("train", [])]
    val_pcwa = [m["pcwa"] for m in ed.get("metrics", {}).get("val", [])]
    preds = ed.get("predictions", [])
    gts = ed.get("ground_truth", [])

    # 1) Loss curves ----------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.title(f"{dset}: Loss vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # 2) PCWA curves ----------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_pcwa, label="Train PCWA")
        plt.plot(epochs, val_pcwa, label="Val PCWA")
        plt.title(f"{dset}: Pattern-Complexity WA vs Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("PCWA")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset}_pcwa_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating PCWA plot for {dset}: {e}")
        plt.close()

    # 3) Confusion matrix ----------------------------------------------
    try:
        if preds and gts:
            labels = sorted(set(gts + preds))
            mat = np.zeros((len(labels), len(labels)), dtype=int)
            lab2idx = {lab: i for i, lab in enumerate(labels)}
            for t, p in zip(gts, preds):
                mat[lab2idx[t], lab2idx[p]] += 1

            plt.figure()
            im = plt.imshow(mat, cmap="Blues")
            plt.colorbar(im)
            plt.title(f"{dset}: Confusion Matrix (Val Set)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(range(len(labels)), labels)
            plt.yticks(range(len(labels)), labels)
            fname = os.path.join(working_dir, f"{dset}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

print("Plotting complete.")
