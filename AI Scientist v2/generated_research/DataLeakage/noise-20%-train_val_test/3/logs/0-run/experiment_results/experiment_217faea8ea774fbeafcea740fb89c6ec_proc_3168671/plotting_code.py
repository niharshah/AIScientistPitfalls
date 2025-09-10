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

# ------------- per-dataset plots ------------------
f1_across_sets = {}
for ds_name, ed in experiment_data.items():
    metrics = ed.get("metrics", {})
    epochs = ed.get("epochs", [])
    preds = np.array(ed.get("predictions", []))
    gts = np.array(ed.get("ground_truth", []))

    # --- 1. loss curves ---
    try:
        plt.figure()
        plt.plot(epochs, metrics.get("train_loss", []), label="Train Loss")
        plt.plot(epochs, metrics.get("val_loss", []), label="Validation Loss")
        plt.title(f"{ds_name} Loss Curves\nLeft: Train, Right: Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {ds_name}: {e}")
        plt.close()

    # --- 2. validation F1 curve ---
    try:
        plt.figure()
        plt.plot(epochs, metrics.get("val_f1", []), marker="o")
        plt.title(f"{ds_name} Validation Macro-F1 Across Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{ds_name}_val_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve for {ds_name}: {e}")
        plt.close()

    # --- 3. confusion matrix ---
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
            fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {ds_name}: {e}")
            plt.close()

    # save final f1 for comparison plot
    if preds.size and gts.size:
        f1_value = f1_score(gts, preds, average="macro")
        f1_across_sets[ds_name] = f1_value
        print(f"{ds_name} Final Test Macro-F1: {f1_value:.4f}")

# ------------- comparison plot (if >1 dataset) -------------
if len(f1_across_sets) > 1:
    try:
        plt.figure()
        names = list(f1_across_sets.keys())
        values = [f1_across_sets[n] for n in names]
        plt.bar(names, values)
        plt.ylim(0, 1)
        plt.title("Validation Macro-F1 Comparison Across Datasets")
        plt.ylabel("Macro-F1")
        plt.tight_layout()
        fname = os.path.join(working_dir, "datasets_f1_comparison.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        plt.close()
