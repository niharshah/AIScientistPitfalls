import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["weight_decay"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = {}


# helper to parse float value from key 'wd_1e-3'
def _wd_float(k):
    try:
        return float(k.split("_", 1)[1])
    except Exception:
        return np.nan


epochs = range(1, 1 + max(len(v["losses"]["train"]) for v in spr_data.values()))

# ---------- figure 1: loss curves ----------
try:
    plt.figure(figsize=(7, 4))
    for k, v in spr_data.items():
        plt.plot(epochs, v["losses"]["train"], "--", label=f"{_wd_float(k):g} train")
        plt.plot(epochs, v["losses"]["val"], "-", label=f"{_wd_float(k):g} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR – Train vs Validation Loss\nLeft: dashed=train, solid=val")
    plt.legend(fontsize=6)
    fname = os.path.join(working_dir, "SPR_loss_curves_weight_decay.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- figure 2: accuracy curves ----------
try:
    plt.figure(figsize=(7, 4))
    for k, v in spr_data.items():
        acc = [m["acc"] for m in v["metrics"]["val"]]
        plt.plot(epochs, acc, label=f"{_wd_float(k):g}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("SPR – Validation Accuracy\nOne line per weight decay")
    plt.legend(title="weight_decay", fontsize=6)
    fname = os.path.join(working_dir, "SPR_val_accuracy_curves.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- figure 3: final acc vs weight decay ----------
try:
    plt.figure(figsize=(5, 4))
    x, y = [], []
    for k, v in spr_data.items():
        x.append(_wd_float(k))
        y.append(v["metrics"]["val"][-1]["acc"])
    plt.scatter(x, y)
    for xi, yi in zip(x, y):
        plt.text(xi, yi, f"{xi:g}", fontsize=7, ha="right")
    plt.xscale("log")
    plt.xlabel("Weight Decay (log scale)")
    plt.ylabel("Final Val Accuracy")
    plt.title("SPR – Final Validation Accuracy vs Weight Decay")
    fname = os.path.join(working_dir, "SPR_final_acc_vs_weight_decay.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating scatter plot: {e}")
    plt.close()

# ---------- print summary ----------
if spr_data:
    print("\nFinal Validation Accuracies:")
    for k, v in sorted(spr_data.items(), key=lambda x: _wd_float(x[0])):
        print(f"wd={_wd_float(k):g} -> acc={v['metrics']['val'][-1]['acc']:.3f}")
