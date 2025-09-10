import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score

# ------------------ paths & data ------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# store val-F1 for cross-dataset comparison
compare_epochs, compare_f1 = {}, {}

for ds_name, ds in experiment_data.items():
    if not isinstance(ds, dict) or "metrics" not in ds:
        continue

    metrics = ds.get("metrics", {})
    epochs = ds.get("epochs", list(range(len(metrics.get("train_loss", [])))))
    preds = np.array(ds.get("predictions", []))
    gts = np.array(ds.get("ground_truth", []))

    # ----- plot 1: loss curves -----
    try:
        plt.figure()
        plt.plot(epochs, metrics.get("train_loss", []), label="Train Loss")
        plt.plot(epochs, metrics.get("val_loss", []), label="Validation Loss")
        plt.title(f"{ds_name} Loss Curves\nLeft: Train Loss, Right: Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {ds_name}: {e}")
        plt.close()

    # ----- plot 2: validation F1 -----
    try:
        v_f1 = metrics.get("val_f1", [])
        plt.figure()
        plt.plot(epochs, v_f1, marker="o")
        plt.title(f"{ds_name} Validation Macro-F1 Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_val_f1_curve.png"))
        plt.close()
        if v_f1:
            compare_epochs[ds_name] = epochs
            compare_f1[ds_name] = v_f1
    except Exception as e:
        print(f"Error creating F1 curve for {ds_name}: {e}")
        plt.close()

    # ----- plot 3: confusion matrix -----
    if preds.size and gts.size:
        try:
            cm = confusion_matrix(gts, preds)
            plt.figure()
            im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.title(
                f"{ds_name} Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
            )
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {ds_name}: {e}")
            plt.close()

    # ----- print final metric -----
    if preds.size and gts.size:
        print(
            f"{ds_name} Final Test Macro-F1: {f1_score(gts, preds, average='macro'):.4f}"
        )

# ----- comparison plot across datasets (max 5) -----
try:
    if compare_f1:
        plt.figure()
        for i, (ds_name, f1_curve) in enumerate(compare_f1.items()):
            if i >= 5:
                break
            plt.plot(compare_epochs[ds_name], f1_curve, marker="o", label=ds_name)
        plt.title("Validation Macro-F1 Comparison Across Datasets")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "datasets_val_f1_comparison.png"))
        plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()
