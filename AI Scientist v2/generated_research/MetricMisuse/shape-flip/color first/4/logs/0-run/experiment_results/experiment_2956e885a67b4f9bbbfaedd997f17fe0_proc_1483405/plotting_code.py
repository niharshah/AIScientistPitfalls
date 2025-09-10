import matplotlib.pyplot as plt
import numpy as np
import os

# setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tags = ["baseline", "uniform_node_feature"]
ds_name = "SPR"


# helper to get arrays safely
def get_list(exp, key1, key2):
    return experiment_data.get(exp, {}).get(ds_name, {}).get(key1, {}).get(key2, [])


# 1 & 2: loss curves
for tag in tags:
    try:
        tr_loss = get_list(tag, "losses", "train")
        val_loss = get_list(tag, "losses", "val")
        if not tr_loss or not val_loss:
            raise ValueError("Loss arrays empty")
        plt.figure()
        plt.plot(tr_loss, label="Train")
        plt.plot(val_loss, label="Validation")
        plt.title(f"{ds_name} {tag} – Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = f"{ds_name}_{tag}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {tag}: {e}")
        plt.close()

# 3 & 4: metric curves (CWA, SWA, CompWA)
for tag in tags:
    try:
        metrics = experiment_data[tag][ds_name]["metrics"]["val"]
        if not metrics:
            raise ValueError("Metric list empty")
        cwa = [m["CWA"] for m in metrics]
        swa = [m["SWA"] for m in metrics]
        comp = [m["CompWA"] for m in metrics]
        plt.figure()
        plt.plot(cwa, label="CWA")
        plt.plot(swa, label="SWA")
        plt.plot(comp, label="CompWA")
        plt.title(f"{ds_name} {tag} – Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.legend()
        fname = f"{ds_name}_{tag}_val_metrics.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {tag}: {e}")
        plt.close()

# 5: bar chart of test metrics comparison
try:
    labels = ["CWA", "SWA", "CompWA"]
    width = 0.35
    x = np.arange(len(labels))
    plt.figure()
    for i, tag in enumerate(tags):
        test_m = experiment_data[tag][ds_name]["metrics"]["test"]
        vals = [test_m["CWA"], test_m["SWA"], test_m["CompWA"]]
        plt.bar(x + i * width, vals, width, label=tag)
    plt.title(f"{ds_name} – Test Metric Comparison")
    plt.xticks(x + width / 2, labels)
    plt.ylabel("Weighted Accuracy")
    plt.legend()
    fname = f"{ds_name}_test_metric_comparison.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating test comparison plot: {e}")
    plt.close()

# print final test metrics
for tag in tags:
    try:
        print(tag, experiment_data[tag][ds_name]["metrics"]["test"])
    except Exception as e:
        print(f"Error printing test metrics for {tag}: {e}")
