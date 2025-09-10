import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
records = experiment_data.get(dataset, {})
model_names = list(records.keys())


# Helper to pull series
def get_series(rec, key, split):
    return [d[key] for d in rec[split][split_type] for split_type in ["train", "val"]]


# ---- 1) Loss curves -----------------------------------------------------------
try:
    plt.figure()
    for name, ls in zip(model_names, ["-", "--", "-."]):
        rec = records[name]
        epochs = [d["epoch"] for d in rec["losses"]["train"]]
        tr = [d["loss"] for d in rec["losses"]["train"]]
        val = [d["loss"] for d in rec["losses"]["val"]]
        plt.plot(epochs, tr, ls, label=f"{name} – train")
        plt.plot(epochs, val, ls, label=f"{name} – val", marker="o")
    plt.title("SPR_BENCH: Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---- 2) Macro-F1 curves -------------------------------------------------------
try:
    plt.figure()
    for name, ls in zip(model_names, ["-", "--", "-."]):
        rec = records[name]
        epochs = [d["epoch"] for d in rec["metrics"]["train"]]
        tr = [d["macro_f1"] for d in rec["metrics"]["train"]]
        val = [d["macro_f1"] for d in rec["metrics"]["val"]]
        plt.plot(epochs, tr, ls, label=f"{name} – train")
        plt.plot(epochs, val, ls, label=f"{name} – val", marker="o")
    plt.title("SPR_BENCH: Macro-F1 vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating macro-F1 curves: {e}")
    plt.close()

# ---- 3) Final Macro-F1 bar chart ---------------------------------------------
try:
    final_f1 = [records[n]["metrics"]["val"][-1]["macro_f1"] for n in model_names]
    plt.figure()
    plt.bar(model_names, final_f1, color=["steelblue", "salmon", "seagreen"])
    for i, v in enumerate(final_f1):
        plt.text(i, v + 0.005, f"{v:.2f}", ha="center")
    plt.title("SPR_BENCH: Final Validation Macro-F1 by Model")
    plt.ylabel("Macro-F1")
    fname = os.path.join(working_dir, "SPR_BENCH_final_macroF1_bars.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ---- 4) Confusion matrix for best model --------------------------------------
try:
    # pick best by last-epoch val F1
    best_idx = int(np.argmax(final_f1))
    best_name = model_names[best_idx]
    best_rec = records[best_name]
    preds = best_rec["predictions"]
    gts = best_rec["ground_truth"]
    cm = confusion_matrix(gts, preds)
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title(f"SPR_BENCH: Confusion Matrix – {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fname = os.path.join(working_dir, f"SPR_BENCH_confusion_{best_name}.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---- Print metrics so they are visible ---------------------------------------
for n, f1 in zip(model_names, final_f1):
    print(f"{n}: final val Macro-F1 = {f1:.3f}")
