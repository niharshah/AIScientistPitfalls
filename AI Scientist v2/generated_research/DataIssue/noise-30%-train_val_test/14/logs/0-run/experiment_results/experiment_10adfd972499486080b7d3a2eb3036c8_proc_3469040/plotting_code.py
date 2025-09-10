import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------
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

# gather per-model statistics
epochs, loss_tr, loss_val, f1_tr, f1_val, final_val_f1 = {}, {}, {}, {}, {}, {}
for model_name, run in runs.items():
    loss_tr[model_name] = [x["loss"] for x in run["losses"]["train"]]
    loss_val[model_name] = [x["loss"] for x in run["losses"]["val"]]
    f1_tr[model_name] = [x["macro_f1"] for x in run["metrics"]["train"]]
    f1_val[model_name] = [x["macro_f1"] for x in run["metrics"]["val"]]
    epochs[model_name] = [x["epoch"] for x in run["metrics"]["train"]]
    final_val_f1[model_name] = f1_val[model_name][-1] if f1_val[model_name] else 0.0

# 1) Loss curves
try:
    plt.figure()
    for m in runs:
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

# 2) Macro-F1 curves
try:
    plt.figure()
    for m in runs:
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

# 3) Bar chart of final val Macro-F1
try:
    plt.figure()
    models = list(runs.keys())
    vals = [final_val_f1[m] for m in models]
    plt.bar(np.arange(len(models)), vals, tick_label=models)
    plt.ylabel("Final Val Macro-F1")
    plt.xlabel("Model")
    plt.title("Final Validation Macro-F1 per Model\nDataset: SPR_BENCH")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_val_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# 4) Confusion matrix for best model
best_model = max(final_val_f1, key=final_val_f1.get) if final_val_f1 else None
if best_model:
    try:
        preds = runs[best_model]["predictions"]
        gts = runs[best_model]["ground_truth"]
        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title(f"Confusion Matrix\nDataset: SPR_BENCH; Model: {best_model}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(working_dir, f"SPR_BENCH_confusion_{best_model}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

# ------------------------------------------------------------------
if final_val_f1:
    print(f"Best model: {best_model} ({final_val_f1[best_model]:.4f} Macro-F1)")
