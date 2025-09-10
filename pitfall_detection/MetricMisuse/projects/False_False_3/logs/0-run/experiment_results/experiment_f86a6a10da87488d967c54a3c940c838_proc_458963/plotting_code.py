import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths & loading ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


def accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


best_dataset, best_acc = None, -1
final_scores = {}

for dset, rec in experiment_data.items():
    # -------------- PLOT 1: loss curves --------------
    try:
        plt.figure()
        tr_loss = rec.get("losses", {}).get("train", [])
        val_loss = rec.get("losses", {}).get("val", [])
        if tr_loss and val_loss:
            plt.plot(tr_loss, "--o", label="train")
            plt.plot(val_loss, "-s", label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dset} Training vs Validation Loss\nType: Sequence Reasoning")
            plt.legend()
            plt.tight_layout()
            fname = f"{dset}_loss_curves.png"
            plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dset}: {e}")
        plt.close()

    # -------------- PLOT 2: validation metric curves (e.g. SWA) --------------
    try:
        plt.figure()
        val_metrics = rec.get("metrics", {}).get("val", [])
        if val_metrics:
            plt.plot(val_metrics, "-d", color="tab:green")
            plt.xlabel("Epoch")
            plt.ylabel("Metric Value")
            plt.title(
                f"{dset} Validation Metric Curve (e.g. SWA)\nType: Sequence Reasoning"
            )
            plt.tight_layout()
            fname = f"{dset}_val_metric_curve.png"
            plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating metric curve for {dset}: {e}")
        plt.close()

    # -------------- compute accuracy & keep best --------------
    y_true = rec.get("ground_truth", [])
    y_pred = rec.get("predictions", [])
    acc = accuracy(y_true, y_pred)
    final_scores[dset] = acc
    print(f"{dset}: final accuracy = {acc:.4f}")
    if acc > best_acc:
        best_acc, best_dataset = acc, dset

# -------------- PLOT 3: bar chart of final accuracies --------------
try:
    plt.figure()
    labels = list(final_scores.keys())
    scores = [final_scores[k] for k in labels]
    x = np.arange(len(labels))
    plt.bar(x, scores, color="tab:purple")
    plt.ylabel("Accuracy")
    plt.title("Final Test Accuracy Comparison\nDatasets")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "datasets_accuracy_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy bar chart: {e}")
    plt.close()

# -------------- PLOT 4: confusion matrix for best dataset --------------
try:
    import itertools

    best_rec = experiment_data.get(best_dataset, {})
    gt = best_rec.get("ground_truth", [])
    pr = best_rec.get("predictions", [])
    labels_set = sorted(set(gt))
    idx = {l: i for i, l in enumerate(labels_set)}
    cm = np.zeros((len(labels_set), len(labels_set)), dtype=int)
    for t, p in zip(gt, pr):
        cm[idx[t], idx[p]] += 1
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title(
        f"{best_dataset} Confusion Matrix (Best Accuracy)\nType: Sequence Reasoning"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels_set)), labels_set, rotation=90, fontsize=6)
    plt.yticks(range(len(labels_set)), labels_set, fontsize=6)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            fontsize=5,
        )
    plt.tight_layout()
    fname = f"{best_dataset}_confusion_matrix.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
