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


# ---------- helper to compute macro-F1 if preds exist ----------
def safe_f1(preds, gts):
    try:
        from sklearn.metrics import f1_score

        return f1_score(gts, preds, average="macro")
    except Exception as e:
        print(f"Could not compute F1: {e}")
        return None


# ---------- iterate over datasets ----------
for dset_name, dset_dict in experiment_data.items():

    # 1) Loss curves ---------------------------------------------------------
    try:
        train_loss = dset_dict.get("losses", {}).get("train", [])
        val_loss = dset_dict.get("losses", {}).get("val", [])
        if train_loss and val_loss:
            plt.figure()
            plt.plot(train_loss, label="Train Loss")
            plt.plot(val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset_name}: Train vs Val Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_loss_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset_name}: {e}")
        plt.close()

    # 2) F1 curves -----------------------------------------------------------
    try:
        tr_f1 = dset_dict.get("metrics", {}).get("train_f1", [])
        val_f1 = dset_dict.get("metrics", {}).get("val_f1", [])
        if tr_f1 and val_f1:
            plt.figure()
            plt.plot(tr_f1, label="Train Macro-F1")
            plt.plot(val_f1, label="Val Macro-F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{dset_name}: Train vs Val Macro-F1")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_name}_f1_curve.png")
            plt.savefig(fname)
            plt.close()

            best_val_f1 = max(val_f1)
            print(f"{dset_name} best validation Macro-F1: {best_val_f1:.4f}")
    except Exception as e:
        print(f"Error creating F1 plot for {dset_name}: {e}")
        plt.close()

    # 3) Confusion matrix (single plot) --------------------------------------
    try:
        preds = np.array(dset_dict.get("predictions", []))
        gts = np.array(dset_dict.get("ground_truth", []))
        if preds.size and gts.size:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(gts, preds)
            plt.figure()
            im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im)
            plt.title(f"{dset_name}: Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            fname = os.path.join(working_dir, f"{dset_name}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()

            test_f1 = safe_f1(preds, gts)
            if test_f1 is not None:
                print(f"{dset_name} test Macro-F1: {test_f1:.4f}")
    except Exception as e:
        print(f"Error creating confusion matrix for {dset_name}: {e}")
        plt.close()
