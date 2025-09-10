import matplotlib.pyplot as plt
import numpy as np
import os

# setup paths
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
    log = experiment_data["remove_color_features"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit(1)

epochs = log["epochs"]
train_loss = log["losses"]["train"]
dev_loss = log["losses"]["dev"]
train_pha = log["metrics"]["train_PHA"]
dev_pha = log["metrics"]["dev_PHA"]
pred = np.asarray(log["predictions"])
gt = np.asarray(log["ground_truth"])
test_metrics = log.get("test_metrics", {})

# 1) Loss curve
try:
    plt.figure()
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, dev_loss, label="Dev")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve - spr_bench (Remove Color Features)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) PHA curve
try:
    plt.figure()
    plt.plot(epochs, train_pha, label="Train PHA")
    plt.plot(epochs, dev_pha, label="Dev PHA")
    plt.xlabel("Epoch")
    plt.ylabel("PHA")
    plt.title("PHA Curve - spr_bench (Remove Color Features)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_pha_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating PHA curve: {e}")
    plt.close()

# 3) Test metric bars
try:
    plt.figure()
    names = list(test_metrics.keys())
    vals = [test_metrics[k] for k in names]
    plt.bar(names, vals, color=["steelblue", "orange", "green"])
    plt.ylim(0, 1)
    plt.title("Test Metrics - spr_bench")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fname = os.path.join(working_dir, "spr_bench_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric bar chart: {e}")
    plt.close()

# 4) Confusion matrix
try:
    n_cls = int(max(gt.max(), pred.max())) + 1
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for g, p in zip(gt, pred):
        cm[g, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - spr_bench")
    plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# print numerical results
print("Test metrics:")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")
