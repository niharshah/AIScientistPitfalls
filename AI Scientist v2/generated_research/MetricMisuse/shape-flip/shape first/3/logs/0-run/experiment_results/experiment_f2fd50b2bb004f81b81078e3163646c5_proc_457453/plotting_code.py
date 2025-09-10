import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------- paths & loading -----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    runs = experiment_data.get("epochs", {})
    # Collect final test metrics and identify best run (by HWA)
    final_scores = {}
    best_run, best_hwa = None, -1
    for k, v in runs.items():
        tm = v["test_metrics"]
        final_scores[k] = tm
        if tm["HWA"] > best_hwa:
            best_hwa, best_run = tm["HWA"], k
        print(
            f"{k}: loss={tm['loss']:.4f}, SWA={tm['SWA']:.4f}, "
            f"CWA={tm['CWA']:.4f}, HWA={tm['HWA']:.4f}"
        )

    # ---------- Figure 1: loss curves ----------
    try:
        plt.figure()
        for k, v in runs.items():
            plt.plot(v["losses"]["train"], "--", label=f"{k} train")
            plt.plot(v["losses"]["val"], "-", label=f"{k} val")
        plt.title("SPR_BENCH Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend(fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ---------- Figure 2: validation HWA ----------
    try:
        plt.figure()
        for k, v in runs.items():
            hwa = [m["HWA"] for m in v["metrics"]["val"]]
            plt.plot(hwa, label=k)
        plt.title("SPR_BENCH Validation Harmonic Weighted Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("HWA")
        plt.legend(fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(working_dir, "SPR_BENCH_val_HWA.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating HWA plot: {e}")
        plt.close()

    # ---------- Figure 3: bar chart of final test metrics ----------
    try:
        plt.figure()
        labels = list(final_scores.keys())
        x = np.arange(len(labels))
        width = 0.25
        swa_vals = [final_scores[k]["SWA"] for k in labels]
        cwa_vals = [final_scores[k]["CWA"] for k in labels]
        hwa_vals = [final_scores[k]["HWA"] for k in labels]
        plt.bar(x - width, swa_vals, width, label="SWA")
        plt.bar(x, cwa_vals, width, label="CWA")
        plt.bar(x + width, hwa_vals, width, label="HWA")
        plt.title("SPR_BENCH Final Test Metrics")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(working_dir, "SPR_BENCH_test_metric_bars.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating bar plot: {e}")
        plt.close()

    # ---------- Figure 4: confusion matrix of best run ----------
    try:
        import itertools
        from collections import Counter

        best_pred = runs[best_run]["predictions"]
        best_gt = runs[best_run]["ground_truth"]
        labels_set = sorted(set(best_gt))
        idx = {l: i for i, l in enumerate(labels_set)}
        cm = np.zeros((len(labels_set), len(labels_set)), dtype=int)
        for t, p in zip(best_gt, best_pred):
            cm[idx[t], idx[p]] += 1

        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title(f"SPR_BENCH Confusion Matrix ({best_run})")
        plt.xticks(range(len(labels_set)), labels_set, rotation=90)
        plt.yticks(range(len(labels_set)), labels_set)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        # annotate cells
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=6,
            )
        plt.tight_layout()
        save_path = os.path.join(working_dir, f"SPR_BENCH_confusion_{best_run}.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
