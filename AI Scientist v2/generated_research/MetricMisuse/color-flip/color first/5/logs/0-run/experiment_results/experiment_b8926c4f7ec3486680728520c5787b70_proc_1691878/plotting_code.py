import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# pick the only ablation/dataset we have
abl = "TokenOrderShuffle"
dset = "SPR_BENCH"
d = experiment_data.get(abl, {}).get(dset, {})

loss_train = d.get("losses", {}).get("train", [])
loss_val = d.get("losses", {}).get("val", [])
metrics_val = d.get("metrics", {}).get("val", [])
metrics_test = d.get("metrics", {}).get("test", {})
preds = np.array(d.get("predictions", []))
gts = np.array(d.get("ground_truth", []))

epochs = list(range(1, len(loss_train) + 1))

# -------- 1. loss curve --------
try:
    plt.figure()
    plt.plot(epochs, loss_train, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curve")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# -------- 2. validation metric curves --------
try:
    if metrics_val:
        acc = [m["acc"] for m in metrics_val]
        cwa = [m["cwa"] for m in metrics_val]
        swa = [m["swa"] for m in metrics_val]
        ccwa = [m["ccwa"] for m in metrics_val]
        plt.figure()
        plt.plot(epochs, acc, label="ACC")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, ccwa, label="CCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH Validation Metrics")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_validation_metrics.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# -------- 3. test metric bar chart --------
try:
    if metrics_test:
        names = ["ACC", "CWA", "SWA", "CCWA"]
        vals = [metrics_test.get(k.lower(), 0) for k in names]
        plt.figure()
        plt.bar(names, vals, color="skyblue")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Test Metrics")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
    plt.close()

# -------- 4. confusion matrix --------
try:
    if preds.size and gts.size and preds.shape == gts.shape:
        n_cls = len(np.unique(np.concatenate([preds, gts])))
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH Confusion Matrix")
        for i in range(n_cls):
            for j in range(n_cls):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print("Plotting complete. Files saved to", working_dir)
