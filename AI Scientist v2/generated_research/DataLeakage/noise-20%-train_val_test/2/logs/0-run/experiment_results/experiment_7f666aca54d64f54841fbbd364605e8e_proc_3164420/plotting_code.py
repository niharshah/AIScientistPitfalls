import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

# ------------ paths -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data, exp_run = {}, {}
else:
    exp_run = experiment_data.get("run", {})

epochs = exp_run.get("epochs", [])
tr_f1 = exp_run.get("metrics", {}).get("train_macro_f1", [])
val_f1 = exp_run.get("metrics", {}).get("val_macro_f1", [])
tr_loss = exp_run.get("losses", {}).get("train", [])
val_loss = exp_run.get("losses", {}).get("val", [])
preds = np.array(exp_run.get("predictions", []))
gts = np.array(exp_run.get("ground_truth", []))
test_macro = exp_run.get("test_macro_f1", np.nan)

# ------------ 1) F1 curves ----------
try:
    plt.figure()
    plt.plot(epochs, tr_f1, "--", label="Train")
    plt.plot(epochs, val_f1, "-", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curves\nLeft: Train dashed, Right: Validation solid")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_macro_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# ------------ 2) Loss curves ---------
try:
    plt.figure()
    plt.plot(epochs, tr_loss, "--", label="Train")
    plt.plot(epochs, val_loss, "-", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curves\nLeft: Train dashed, Right: Validation solid")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------ 3) Confusion matrix ----
try:
    if preds.size and gts.size:
        cm = confusion_matrix(gts, preds)
        disp = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        plt.title("SPR_BENCH Test Confusion Matrix")
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------ 4) Per-class F1 bar ----
try:
    if preds.size and gts.size:
        cls_f1 = f1_score(gts, preds, average=None)
        plt.figure()
        plt.bar(np.arange(len(cls_f1)), cls_f1)
        plt.xlabel("Class ID")
        plt.ylabel("F1 Score")
        plt.title("SPR_BENCH Per-Class F1 Scores (Test Set)")
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_per_class_f1.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating per-class F1 plot: {e}")
    plt.close()

# ------------ print summary ----------
print(f"Test Macro-F1: {test_macro:.4f}")
if preds.size and gts.size:
    print("Per-class F1:", f1_score(gts, preds, average=None))
