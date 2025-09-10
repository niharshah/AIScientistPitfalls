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

runs = experiment_data.get("weight_decay_tuning", {})

# 1) Train / Val loss curves -------------------------------------------------
try:
    plt.figure()
    for key, run in runs.items():
        tr = np.array(run["losses"]["train"])  # (epoch, loss)
        vl = np.array(run["losses"]["val"])
        if tr.size:
            plt.plot(tr[:, 0], tr[:, 1], label=f"train {key}")
        if vl.size:
            plt.plot(vl[:, 0], vl[:, 1], "--", label=f"val {key}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("SPR synthetic – Train/Val Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, "spr_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) Validation CSHM curves --------------------------------------------------
try:
    plt.figure()
    for key, run in runs.items():
        m = np.array(run["metrics"]["val"])  # (epoch, cwa, swa, cshm)
        if m.size:
            plt.plot(m[:, 0], m[:, 3], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("CSHM")
    plt.title("SPR synthetic – Validation CSHM")
    plt.legend()
    fname = os.path.join(working_dir, "spr_val_cshm_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating CSHM curves: {e}")
    plt.close()


# 3) Final test accuracy bar chart ------------------------------------------
def simple_acc(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    return (pred == gt).mean() if gt.size else 0.0


try:
    labels, accs = [], []
    for key, run in runs.items():
        labels.append(key)
        accs.append(simple_acc(run.get("predictions", []), run.get("ground_truth", [])))
    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, accs, color="skyblue")
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Accuracy")
    plt.title("SPR synthetic – Test Accuracy per Weight Decay")
    fname = os.path.join(working_dir, "spr_test_accuracy.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy bar chart: {e}")
    plt.close()
