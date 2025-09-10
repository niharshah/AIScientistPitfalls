import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helpers ----------
def count_color_variety(seq):
    return len(set(tok[1:] for tok in seq.split()))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def comp_weight(seq):
    return count_color_variety(seq) + count_shape_variety(seq)


# ---------- per-dataset plots ----------
val_acc_by_ds = {}
for ds_name, ds in experiment_data.items():
    epochs = np.arange(1, len(ds["losses"]["train"]) + 1)
    train_loss = ds["losses"]["train"]
    val_loss = ds["losses"]["val"]
    train_acc = [m["acc"] for m in ds["metrics"]["train"]]
    val_acc = [m["acc"] for m in ds["metrics"]["val"]]
    val_cwa = [m["compWA"] for m in ds["metrics"]["val"]]
    val_acc_by_ds[ds_name] = val_acc

    # ---- 1. loss curves ----
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # ---- 2. accuracy curves ----
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} Accuracy Curves\nLeft: Train, Right: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()

    # ---- 3. CompWA curve ----
    try:
        plt.figure()
        plt.plot(epochs, val_cwa, label="Validation CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        plt.title(f"{ds_name} Complexity-Weighted Accuracy\nValidation Only")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_compwa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot for {ds_name}: {e}")
        plt.close()

    # ---- 4. Confusion matrix (optional) ----
    try:
        preds = np.array(ds.get("predictions", []))
        gts = np.array(ds.get("ground_truth", []))
        if preds.size and gts.size:
            num_classes = int(max(gts.max(), preds.max()) + 1)
            if num_classes <= 15:  # keep readable
                cm = np.zeros((num_classes, num_classes), dtype=int)
                for p, t in zip(preds, gts):
                    cm[t, p] += 1
                plt.figure(figsize=(5, 4))
                plt.imshow(cm, cmap="Blues")
                plt.colorbar()
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"{ds_name} Confusion Matrix")
                plt.savefig(
                    os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
                )
                plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        plt.close()

    # ---- 5. metrics print ----
    try:
        if preds.size and gts.size:
            weights = np.array([comp_weight(s) for s in ds["sequences"]])
            acc = (preds == gts).mean()
            cwa = (weights * (preds == gts)).sum() / weights.sum()
            print(f"{ds_name} -- Test Accuracy: {acc:.3f} | Test CompWA: {cwa:.3f}")
    except Exception as e:
        print(f"Error computing metrics for {ds_name}: {e}")

# ---------- cross-dataset comparison (if >1) ----------
if len(val_acc_by_ds) > 1:
    try:
        plt.figure()
        for ds_name, accs in val_acc_by_ds.items():
            plt.plot(np.arange(1, len(accs) + 1), accs, label=ds_name)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy Comparison Across Datasets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "datasets_val_accuracy_comparison.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating cross-dataset comparison plot: {e}")
        plt.close()
