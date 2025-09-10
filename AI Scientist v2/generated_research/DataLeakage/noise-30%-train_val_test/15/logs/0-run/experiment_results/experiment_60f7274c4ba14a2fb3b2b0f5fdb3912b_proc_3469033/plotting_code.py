import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- Load experiment data -----------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

final_val_f1 = {}  # collect for cross-dataset comparison

for ds_name, ds_dict in experiment_data.items():
    epochs = ds_dict.get("epochs", [])
    tr_loss = ds_dict["losses"]["train"] if "losses" in ds_dict else []
    val_loss = ds_dict["losses"]["val"] if "losses" in ds_dict else []
    tr_f1 = ds_dict["metrics"]["train"] if "metrics" in ds_dict else []
    val_f1 = ds_dict["metrics"]["val"] if "metrics" in ds_dict else []
    preds = ds_dict.get("predictions", [])
    gts = ds_dict.get("ground_truth", [])
    test_f1 = ds_dict.get("test_macroF1", None)
    if test_f1 is not None:
        print(f"{ds_name} – Test Macro-F1: {test_f1:.4f}")

    # save last val f1 for comparison
    if val_f1:
        final_val_f1[ds_name] = val_f1[-1]

    # ---------------- Loss curve ---------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name}: Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting loss for {ds_name}: {e}")
        plt.close()

    # ---------------- Macro-F1 curve -----------------------------------
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{ds_name}: Training vs Validation Macro-F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_macroF1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error plotting f1 for {ds_name}: {e}")
        plt.close()

    # ---------------- Confusion matrix ---------------------------------
    try:
        if preds and gts:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(gts, preds, normalize="true")
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(
                f"{ds_name}: Normalised Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_conf_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {ds_name}: {e}")
        plt.close()

    # ---------------- Label distribution --------------------------------
    try:
        if gts:
            labels, counts = np.unique(gts, return_counts=True)
            plt.figure()
            plt.bar(labels, counts)
            plt.xlabel("Label")
            plt.ylabel("Frequency")
            plt.title(f"{ds_name}: Ground-Truth Label Distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds_name}_label_distribution.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting label distribution for {ds_name}: {e}")
        plt.close()

# ---------------- Cross-dataset comparison plot -------------------------
try:
    if final_val_f1:
        plt.figure()
        names = list(final_val_f1.keys())
        scores = [final_val_f1[n] for n in names]
        plt.barh(names, scores)
        plt.xlabel("Final Validation Macro-F1")
        plt.title("Comparison of Final Validation Macro-F1 Across Datasets")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "cross_dataset_val_macroF1.png"))
        plt.close()
except Exception as e:
    print(f"Error plotting cross-dataset comparison: {e}")
    plt.close()
