import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
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


# ---------- helper ----------
def safe(arr_dict, key):  # returns np.ndarray (possibly size 0)
    return np.array(arr_dict.get(key, []))


# ---------- iterate over datasets ----------
for dset, dct in experiment_data.items():
    metrics, losses = dct.get("metrics", {}), dct.get("losses", {})

    # 1) Accuracy curves ------------------------------------------------------
    try:
        tr, va = safe(metrics, "train_acc"), safe(metrics, "val_acc")
        if tr.size and va.size:
            plt.figure()
            ep = np.arange(1, len(tr) + 1)
            plt.plot(ep, tr, label="Train Acc")
            plt.plot(ep, va, label="Val Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dset}: Train vs Validation Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dset}_accuracy_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dset}: {e}")
        plt.close()

    # 2) Loss curves ----------------------------------------------------------
    try:
        tl, vl = safe(losses, "train"), safe(losses, "val")
        if tl.size and vl.size:
            plt.figure()
            ep = np.arange(1, len(tl) + 1)
            plt.plot(ep, tl, label="Train Loss")
            plt.plot(ep, vl, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset}: Train vs Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dset}_loss_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # 3) Rule-fidelity curve --------------------------------------------------
    try:
        rf = safe(metrics, "rule_fidelity")
        if rf.size:
            plt.figure()
            ep = np.arange(1, len(rf) + 1)
            plt.plot(ep, rf, label="Rule Fidelity")
            plt.xlabel("Epoch")
            plt.ylabel("Fidelity")
            plt.title(f"{dset}: Rule Fidelity Over Epochs")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dset}_rule_fidelity.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating fidelity plot for {dset}: {e}")
        plt.close()

    # 4) Fidelity vs Val-Accuracy --------------------------------------------
    try:
        rf, va = safe(metrics, "rule_fidelity"), safe(metrics, "val_acc")
        if rf.size and va.size:
            plt.figure()
            ep = np.arange(1, len(rf) + 1)
            plt.plot(ep, va, label="Val Acc")
            plt.plot(ep, rf, label="Rule Fidelity")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy / Fidelity")
            plt.title(f"{dset}: Validation Accuracy vs Rule Fidelity")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dset}_val_vs_fidelity.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating val-vs-fidelity plot for {dset}: {e}")
        plt.close()

    # 5) Confusion matrix (test set) ------------------------------------------
    try:
        preds, gts = safe(dct, "predictions"), safe(dct, "ground_truth")
        if preds.size and gts.size and preds.shape == gts.shape:
            n_cls = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((n_cls, n_cls), int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dset}: Confusion Matrix (Test Set)")
            plt.savefig(os.path.join(working_dir, f"{dset}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()

    # ---------- evaluation metric ----------
    if "predictions" in dct and "ground_truth" in dct:
        preds, gts = np.array(dct["predictions"]), np.array(dct["ground_truth"])
        if preds.size and preds.shape == gts.shape:
            acc = (preds == gts).mean()
            print(f"{dset} – Test Accuracy: {acc:.3f}")
