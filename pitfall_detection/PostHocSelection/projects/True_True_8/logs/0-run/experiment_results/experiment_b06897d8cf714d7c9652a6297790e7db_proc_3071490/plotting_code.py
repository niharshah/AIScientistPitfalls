import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# helper: collect metric arrays for each wd
metric_names = ["train_loss", "val_loss", "val_ACS"]
metrics = {m: {} for m in metric_names}  # e.g. metrics['train_loss'][wd]=([ep],[vals])

for wd, runs in experiment_data.get("weight_decay", {}).items():
    mdict = runs["SPR_BENCH"]["metrics"]
    for m in metric_names:
        ep, vals = zip(*mdict[m]) if mdict[m] else ([], [])
        metrics[m][wd] = (np.array(ep), np.array(vals))

# ------------- PLOT 1–3: curves -----------------
plot_info = [
    ("train_loss", "Training Loss vs Epochs"),
    ("val_loss", "Validation Loss vs Epochs"),
    ("val_ACS", "Validation ACS vs Epochs"),
]

for m, title in plot_info:
    try:
        plt.figure()
        for wd, (ep, vals) in metrics[m].items():
            plt.plot(ep, vals, label=f"wd={wd}")
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.title(f"SPR_BENCH: {title}")
        plt.legend()
        fname = f"SPR_BENCH_{m}_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for {m}: {e}")
        plt.close()

# ------------- PLOT 4: confusion matrix for best run -------------
try:
    # pick wd with lowest final val_loss
    best_wd = None
    best_loss = float("inf")
    for wd, (ep, vals) in metrics["val_loss"].items():
        if len(vals) and vals[-1] < best_loss:
            best_loss = vals[-1]
            best_wd = wd
    if best_wd is not None:
        run = experiment_data["weight_decay"][best_wd]["SPR_BENCH"]
        preds = run["predictions"]
        gts = run["ground_truth"]
        epochs = len(run["metrics"]["val_loss"])
        dev_size = len(gts) // epochs if epochs else 0
        preds = np.array(preds[-dev_size:])
        gts = np.array(gts[-dev_size:])
        num_cls = max(gts.max(), preds.max()) + 1
        cm = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1

        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title(f"SPR_BENCH Confusion Matrix (best wd={best_wd})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black",
                )
        fname = f"SPR_BENCH_confusion_matrix_wd_{best_wd}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
