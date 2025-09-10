import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------ load experiment data -------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# only proceed if expected key exists
key_path = ("Remove_L1_Sparsity", "SPR_BENCH")
if experiment_data.get(key_path[0], {}).get(key_path[1]):
    ed = experiment_data[key_path[0]][key_path[1]]
    losses = ed["losses"]
    metrics = ed["metrics"]
    REA_dev, REA_test = ed.get("REA_dev"), ed.get("REA_test")
    preds, gts = ed.get("predictions"), ed.get("ground_truth")
    epochs = range(1, len(losses["train"]) + 1)

    # ---------- Plot 1: loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ---------- Plot 2: F1 curves ----------
    try:
        plt.figure()
        plt.plot(epochs, metrics["train_f1"], label="Train F1")
        plt.plot(epochs, metrics["val_f1"], label="Val F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH F1 Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # ---------- Plot 3: REA vs Hybrid ----------
    try:
        plt.figure()
        bars = [REA_dev, REA_test, metrics["val_f1"][-1]]
        labels = ["REA_dev", "REA_test", "Hybrid_test_F1"]
        plt.bar(labels, bars, color=["skyblue", "lightgreen", "salmon"])
        plt.ylim(0, 1)
        for i, v in enumerate(bars):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
        plt.title("SPR_BENCH: Rule Accuracy vs Hybrid Model")
        fname = os.path.join(working_dir, "SPR_BENCH_rule_vs_hybrid.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating REA bar chart: {e}")
        plt.close()

    # ---------- Plot 4: Confusion Matrix ----------
    if preds and gts:
        try:
            cm = confusion_matrix(gts, preds)
            # Optional clipping to 10x10 for readability
            if cm.shape[0] > 10:
                cm = cm[:10, :10]
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.title("SPR_BENCH Confusion Matrix (Test)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()

    # ------------- print key metrics -------------
    best_val_f1 = max(metrics["val_f1"]) if metrics["val_f1"] else None
    print(f"Best Val F1: {best_val_f1:.4f}" if best_val_f1 else "Val F1 unavailable")
    print(f"REA_dev: {REA_dev:.4f}  REA_test: {REA_test:.4f}")
    print(
        f"Hybrid Test Macro-F1: {metrics['val_f1'][-1]:.4f}"
        if metrics["val_f1"]
        else ""
    )
else:
    print("Expected experiment entry not found in experiment_data.npy.")
