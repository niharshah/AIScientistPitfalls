import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score

# ------------------- setup & data loading -------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset_name = "SPR_BENCH"
runs = experiment_data.get(dataset_name, {})
models = ["Baseline", "SymbolicAug"]
epochs = {}
loss_tr = {}
loss_val = {}
f1_tr = {}
f1_val = {}
final_val_f1 = {}

for m in models:
    run = runs.get(m, {})
    epochs[m] = [x["epoch"] for x in run.get("metrics", {}).get("train", [])]
    loss_tr[m] = [x["loss"] for x in run.get("losses", {}).get("train", [])]
    loss_val[m] = [x["loss"] for x in run.get("losses", {}).get("val", [])]
    f1_tr[m] = [x["macro_f1"] for x in run.get("metrics", {}).get("train", [])]
    f1_val[m] = [x["macro_f1"] for x in run.get("metrics", {}).get("val", [])]
    final_val_f1[m] = f1_val[m][-1] if f1_val[m] else 0.0

# ------------------- 1) Loss curves -------------------
try:
    plt.figure()
    for m in models:
        if epochs[m]:
            plt.plot(epochs[m], loss_tr[m], "--", label=f"{m}-train")
            plt.plot(epochs[m], loss_val[m], label=f"{m}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training vs Validation Loss\nDataset: SPR_BENCH")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------- 2) Macro-F1 curves -------------------
try:
    plt.figure()
    for m in models:
        if epochs[m]:
            plt.plot(epochs[m], f1_tr[m], "--", label=f"{m}-train")
            plt.plot(epochs[m], f1_val[m], label=f"{m}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training vs Validation Macro-F1\nDataset: SPR_BENCH")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# ------------------- 3) Bar chart of final val F1 -------------------
try:
    plt.figure()
    xs = np.arange(len(models))
    vals = [final_val_f1[m] for m in models]
    plt.bar(xs, vals, tick_label=models)
    plt.xlabel("Model")
    plt.ylabel("Final Val Macro-F1")
    plt.title("Final Validation Macro-F1 per Model\nDataset: SPR_BENCH")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_val_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ------------------- 4-5) Confusion matrices (one per model) -------------------
for m in models:
    try:
        preds = runs[m].get("predictions", [])
        gts = runs[m].get("ground_truth", [])
        if preds and gts:
            cm = confusion_matrix(gts, preds, labels=sorted(set(gts)))
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix\nModel: {m} | Dataset: SPR_BENCH")
            # add counts
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.tight_layout()
            fname = f"SPR_BENCH_confusion_matrix_{m}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {m}: {e}")
        plt.close()

# ------------------- print best model -------------------
if final_val_f1:
    best_model = max(final_val_f1, key=final_val_f1.get)
    print(
        f"Best model by final validation Macro-F1: {best_model} "
        f"({final_val_f1[best_model]:.4f})"
    )
