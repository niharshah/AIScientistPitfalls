import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_exp = experiment_data.get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_exp = {}

epochs = np.array(spr_exp.get("epochs", []))
train_loss = np.array(spr_exp.get("losses", {}).get("train", []))
val_loss = np.array(spr_exp.get("losses", {}).get("val", []))
val_cwa = np.array(spr_exp.get("metrics", {}).get("val_cwa", []))
val_f1 = np.array(spr_exp.get("metrics", {}).get("val_f1", []))
preds = np.array(spr_exp.get("predictions", []))
gts = np.array(spr_exp.get("ground_truth", []))

# 1) Loss curves -------------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss", color="tab:blue")
    plt.plot(
        epochs, val_loss, label="Validation Loss", color="tab:orange", linestyle="--"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Metric curves -----------------------------------------------------------
try:
    plt.figure()
    plt.plot(epochs, val_cwa, label="Val CWA", color="tab:green")
    plt.plot(epochs, val_f1, label="Val Macro-F1", color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Validation CWA & Macro-F1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_metric_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric curves: {e}")
    plt.close()

# 3) Final-epoch bar chart ---------------------------------------------------
try:
    plt.figure()
    bars = ["CWA", "Macro-F1"]
    final_scores = [
        val_cwa[-1] if val_cwa.size else 0,
        val_f1[-1] if val_f1.size else 0,
    ]
    plt.bar(bars, final_scores, color=["tab:green", "tab:red"])
    for x, y in zip(bars, final_scores):
        plt.text(x, y + 0.01, f"{y:.3f}", ha="center", va="bottom")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("SPR_BENCH: Final Validation Scores")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_scores.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final score bar chart: {e}")
    plt.close()

# 4) Confusion matrix --------------------------------------------------------
try:
    cm = np.zeros((2, 2), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
    plt.colorbar()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# -------- optional quick print ----------
if val_cwa.size and val_f1.size:
    print(f"Final Validation CWA: {val_cwa[-1]:.4f}")
    print(f"Final Validation Macro-F1: {val_f1[-1]:.4f}")
