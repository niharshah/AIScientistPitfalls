import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

spr = experiment_data.get("SPR_BENCH", {})

# ----------------------------------------------------------- plot 1: loss curves
try:
    tr_loss = spr["losses"]["train"]  # list[(epoch,val)]
    val_loss = spr["losses"]["val"]
    epochs = [e for e, _ in tr_loss]
    plt.figure()
    plt.plot(epochs, [v for _, v in tr_loss], label="Train Loss")
    plt.plot(epochs, [v for _, v in val_loss], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves (Train vs Val)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# ----------------------------------------------------------- plot 2: comp-WA curves
try:
    tr_acc = spr["metrics"]["train"]
    val_acc = spr["metrics"]["val"]
    epochs = [e for e, _ in tr_acc]
    plt.figure()
    plt.plot(epochs, [v for _, v in tr_acc], label="Train CompWA")
    plt.plot(epochs, [v for _, v in val_acc], label="Val CompWA")
    plt.xlabel("Epoch")
    plt.ylabel("Comp-Weighted Accuracy")
    plt.title("SPR_BENCH CompWA Curves (Train vs Val)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_compWA_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CompWA curve: {e}")
    plt.close()

# ----------------------------------------------------------- plot 3: test metrics bar chart
try:
    test_metrics = spr["metrics"]["test"]  # [CWA,SWA,CompWA]
    metric_names = ["Color-WA", "Shape-WA", "Comp-WA"]
    plt.figure()
    plt.bar(metric_names, test_metrics, color=["skyblue", "salmon", "seagreen"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Test Metrics")
    fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
    plt.close()

# ----------------------------------------------------------- plot 4: confusion matrix
try:
    y_true = np.array(spr["ground_truth"])
    y_pred = np.array(spr["predictions"])
    n_cls = max(y_true.max(), y_pred.max()) + 1
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    plt.figure()
    im = plt.imshow(conf, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SPR_BENCH Confusion Matrix")
    for i in range(n_cls):
        for j in range(n_cls):
            plt.text(
                j,
                i,
                conf[i, j],
                ha="center",
                va="center",
                color="white" if conf[i, j] > conf.max() / 2 else "black",
                fontsize=8,
            )
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ----------------------------------------------------------- print test metrics
if spr.get("metrics", {}).get("test"):
    cwa, swa, comp = spr["metrics"]["test"]
    print(f"Loaded Test Metrics  ->  CWA={cwa:.4f} | SWA={swa:.4f} | CompWA={comp:.4f}")
