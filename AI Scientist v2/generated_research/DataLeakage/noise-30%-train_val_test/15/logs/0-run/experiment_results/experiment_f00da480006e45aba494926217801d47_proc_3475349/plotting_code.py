import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------------------------------------------------------------
exp_name = "Remove_Unigram_Count_Features"
ds_name = "SPR_BENCH"
ed = experiment_data.get(exp_name, {}).get(ds_name, None)
if ed is None:
    print("No data found for the specified experiment/dataset.")
    exit()

epochs = ed.get("epochs", [])
tr_loss = ed["losses"]["train"]
val_loss = ed["losses"]["val"]
tr_f1 = ed["metrics"]["train"]
val_f1 = ed["metrics"]["val"]
preds = np.array(ed.get("predictions", []))
gts = np.array(ed.get("ground_truth", []))
test_loss = ed.get("test_loss", None)
test_f1 = ed.get("test_macroF1", None)

# ---------------- Plot 1: Loss curves --------------------------------
try:
    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{exp_name} – {ds_name} Loss Curves")
    plt.legend()
    fpath = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ---------------- Plot 2: F1 curves ----------------------------------
try:
    plt.figure()
    plt.plot(epochs, tr_f1, label="Train Macro-F1")
    plt.plot(epochs, val_f1, label="Validation Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(f"{exp_name} – {ds_name} Macro-F1 Curves")
    plt.legend()
    fpath = os.path.join(working_dir, f"{ds_name}_f1_curves.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# ---------------- Plot 3: Confusion matrix ---------------------------
try:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(gts, preds)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{exp_name} – {ds_name} Confusion Matrix")
    fpath = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ---------------- Plot 4: Prediction vs GT histogram -----------------
try:
    plt.figure(figsize=(8, 4))
    uniq = np.arange(max(gts.max(), preds.max()) + 1)
    plt.subplot(1, 2, 1)
    gt_counts = [(gts == u).sum() for u in uniq]
    plt.bar(uniq, gt_counts)
    plt.title("Ground Truth Counts")
    plt.subplot(1, 2, 2)
    pred_counts = [(preds == u).sum() for u in uniq]
    plt.bar(uniq, pred_counts, color="orange")
    plt.title("Prediction Counts")
    plt.suptitle(
        f"{exp_name} – {ds_name}\nLeft: Ground Truth, Right: Generated Samples"
    )
    fpath = os.path.join(working_dir, f"{ds_name}_class_count_comparison.png")
    plt.savefig(fpath)
    plt.close()
except Exception as e:
    print(f"Error creating class count plot: {e}")
    plt.close()

# ---------------- Print final metrics --------------------------------
print(f"Test Loss: {test_loss:.4f} | Test Macro-F1: {test_f1:.4f}")
