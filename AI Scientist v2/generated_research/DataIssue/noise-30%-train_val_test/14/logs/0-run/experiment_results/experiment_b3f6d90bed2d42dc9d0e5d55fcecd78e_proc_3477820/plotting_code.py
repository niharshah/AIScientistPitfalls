import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset = "SPR_BENCH"
models = ["Baseline", "SymToken", "FrozenEmb"]
metrics = {m: experiment_data[dataset][m]["metrics"] for m in models}
losses = {m: experiment_data[dataset][m]["losses"] for m in models}


# ---------------- convenience helpers ----------------
def extract(lst, key):
    """lst is list of dicts[{epoch:int, key:float}] -> xs, ys"""
    xs, ys = [], []
    for d in lst:
        xs.append(d["epoch"])
        ys.append(d[key])
    return xs, ys


final_scores = {}

# 1) Loss curves --------------------------------------------------------------
try:
    plt.figure()
    for m in models:
        x_tr, y_tr = extract(losses[m]["train"], "loss")
        x_va, y_va = extract(losses[m]["val"], "loss")
        plt.plot(x_tr, y_tr, label=f"{m}-train")
        plt.plot(x_va, y_va, "--", label=f"{m}-val")
    plt.title("SPR_BENCH: Training/Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Macro-F1 curves ----------------------------------------------------------
try:
    plt.figure()
    for m in models:
        x_va, y_va = extract(metrics[m]["val"], "macro_f1")
        plt.plot(x_va, y_va, label=m)
        final_scores[m] = {"f1": y_va[-1]}  # store latest
    plt.title("SPR_BENCH: Validation macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("macro-F1")
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# 3) Accuracy (RGA) curves ----------------------------------------------------
try:
    plt.figure()
    for m in models:
        x_va, y_va = extract(metrics[m]["val"], "RGA")
        plt.plot(x_va, y_va, label=m)
        final_scores[m]["acc"] = y_va[-1]
    plt.title("SPR_BENCH: Validation Accuracy (RGA)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# 4) Final macro-F1 bar chart -------------------------------------------------
try:
    plt.figure()
    plt.bar(list(final_scores.keys()), [final_scores[m]["f1"] for m in models])
    plt.title("SPR_BENCH: Final Validation macro-F1 by Model")
    plt.ylabel("macro-F1")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_F1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 bar chart: {e}")
    plt.close()

# 5) Confusion matrix of best model ------------------------------------------
try:
    best_model = max(final_scores, key=lambda m: final_scores[m]["f1"])
    preds = np.array(experiment_data[dataset][best_model]["predictions"])
    gts = np.array(experiment_data[dataset][best_model]["ground_truth"])
    num_labels = len(np.unique(np.concatenate([preds, gts])))
    cm = np.zeros((num_labels, num_labels), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.title(f"SPR_BENCH: Confusion Matrix (Dev) – {best_model}")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    ticks = np.arange(num_labels)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.savefig(os.path.join(working_dir, f"SPR_BENCH_confusion_{best_model}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------- print final metrics table ----------------------------------
print("\nFinal Validation Metrics")
for m in models:
    f1 = final_scores[m]["f1"]
    acc = final_scores[m]["acc"]
    print(f"{m:12s}  macro-F1: {f1:.3f}  Accuracy: {acc:.3f}")
