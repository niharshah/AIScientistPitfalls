import matplotlib.pyplot as plt
import numpy as np
import os

# working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data ------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# short-cuts
exp = experiment_data.get("MeanPooling", {}).get("SPR_BENCH", {})
loss_tr = exp.get("losses", {}).get("train", [])
loss_val = exp.get("losses", {}).get("val", [])
cwa_val = exp.get("metrics", {}).get("val", [])
pred = np.array(exp.get("predictions", []))
gt = np.array(exp.get("ground_truth", []))

# ---------------- metrics --------------------
acc = float((pred == gt).mean()) if pred.size else None
best_cwa = cwa_val[-1] if cwa_val else None
print(f"Validation accuracy: {acc:.4f}" if acc is not None else "No predictions found.")
print(
    f"Best Comp-Weighted-Accuracy: {best_cwa:.4f}"
    if best_cwa is not None
    else "No CWA recorded."
)

# ---------------- plots ----------------------
# 1. Loss curves
try:
    plt.figure()
    if loss_tr:
        plt.plot(loss_tr, label="Train")
    if loss_val:
        plt.plot(loss_val, label="Validation")
    plt.title("SPR_BENCH - MeanPooling\nLoss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    if loss_tr or loss_val:
        plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_MeanPooling_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2. CWA curve
try:
    if cwa_val:
        plt.figure()
        plt.plot(cwa_val, marker="o")
        plt.title("SPR_BENCH - MeanPooling\nValidation Comp-Weighted-Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("CWA")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_MeanPooling_CWA_curve.png"))
        plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# 3. Confusion matrix heat-map (if data present)
try:
    if pred.size and gt.size:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(gt, pred, labels=np.unique(np.concatenate([gt, pred])))
        plt.figure(figsize=(5, 5))
        plt.imshow(cm, cmap="Blues")
        plt.title("SPR_BENCH - MeanPooling\nConfusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar(label="Count")
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_MeanPooling_confusion_matrix.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
