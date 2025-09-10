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

for dname, dct in experiment_data.items():
    # ------- loss curves -------
    try:
        tr = np.array(dct["losses"]["train"])  # shape (E,2)
        va = np.array(dct["losses"]["val"])
        plt.figure()
        plt.plot(tr[:, 0], tr[:, 1], label="Train")
        plt.plot(va[:, 0], va[:, 1], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname} Training / Validation Loss\nLeft: Train, Right: Val")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_loss_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dname}: {e}")
        plt.close()

    # ------- validation HCSA metric -------
    try:
        met = np.array(dct["metrics"]["val"])
        plt.figure()
        plt.plot(met[:, 0], met[:, 1], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("HCSA")
        plt.title(f"{dname} Validation Harmonic CSA Across Epochs")
        fname = os.path.join(working_dir, f"{dname}_val_hcsa_curve.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating HCSA curve for {dname}: {e}")
        plt.close()

    # ------- confusion matrix -------
    try:
        preds = np.array(dct["predictions"])
        trues = np.array(dct["ground_truth"])
        labels = sorted(set(preds).union(trues))
        L = len(labels)
        lab2i = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((L, L), dtype=int)
        for t, p in zip(trues, preds):
            cm[lab2i[t], lab2i[p]] += 1
        acc = (preds == trues).mean()
        plt.figure(figsize=(6, 5))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(L), labels, rotation=90)
        plt.yticks(range(L), labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{dname} Confusion Matrix (Acc={acc:.2%})")
        fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()
