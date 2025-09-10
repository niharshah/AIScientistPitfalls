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


# helper to safely extract scalar from 1-element lists
def first(lst, default=np.nan):
    try:
        return lst[0] if isinstance(lst, (list, tuple)) and lst else default
    except Exception:
        return default


# gather data
variants = list(experiment_data.keys())
dataset = "SPR_BENCH"  # only dataset present
val_acc, test_acc, val_loss, depths, rule_lens = [], [], [], [], []
for v in variants:
    info = experiment_data.get(v, {}).get(dataset, {})
    m = info.get("metrics", {})
    l = info.get("losses", {})
    val_acc.append(first(m.get("val", [])))
    test_acc.append(first(m.get("test", [])))
    val_loss.append(first(l.get("val", [])))
    depths.append(first(m.get("rule_depth", [])))
    rule_lens.append(first(m.get("avg_rule_len", [])))

# 1) Accuracy comparison
try:
    x = np.arange(len(variants))
    width = 0.35
    plt.figure()
    plt.bar(x - width / 2, val_acc, width, label="Validation")
    plt.bar(x + width / 2, test_acc, width, label="Test")
    plt.xticks(x, variants, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset}: Validation vs Test Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, f"{dataset}_accuracy_comparison.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) Validation loss comparison
try:
    plt.figure()
    plt.bar(variants, val_loss, color="orange")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Log Loss")
    plt.title(f"{dataset}: Validation Loss")
    fname = os.path.join(working_dir, f"{dataset}_val_loss_comparison.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) Model complexity (depth & avg rule length)
try:
    x = np.arange(len(variants))
    width = 0.35
    fig, ax1 = plt.subplots()
    ax1.bar(x - width / 2, depths, width, label="Tree Depth", color="green")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, rule_lens, width, label="Avg Rule Length", color="purple")
    ax1.set_ylabel("Depth")
    ax2.set_ylabel("Avg Rule Len")
    plt.xticks(x, variants, rotation=45, ha="right")
    plt.title(f"{dataset}: Model Complexity Metrics")
    fig.legend(loc="upper right")
    fname = os.path.join(working_dir, f"{dataset}_complexity_metrics.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating complexity plot: {e}")
    plt.close()

print("Plot generation complete.")
