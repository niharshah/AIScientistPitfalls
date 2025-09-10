import matplotlib.pyplot as plt
import numpy as np
import os

# working directory for plots
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------ #
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment_data.npy: {e}")
    exit(0)


# ------------------------------------------------------------------ #
# helper to fetch nested dict safely
def get(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


ds_name = "no_color_edge"
model_name = "SPR_RGCN"
exp = get(experiment_data, ds_name, model_name, default={})
epochs = exp.get("epochs", [])
loss_tr = get(exp, "losses", "train", default=[])
loss_val = get(exp, "losses", "val", default=[])
cwa_tr = get(exp, "metrics", "train", "CWA", default=[])
cwa_val = get(exp, "metrics", "val", "CWA", default=[])
swa_tr = get(exp, "metrics", "train", "SWA", default=[])
swa_val = get(exp, "metrics", "val", "SWA", default=[])
cpx_tr = get(exp, "metrics", "train", "CmpWA", default=[])
cpx_val = get(exp, "metrics", "val", "CmpWA", default=[])
y_true = exp.get("ground_truth", [])
y_pred = exp.get("predictions", [])

# ------------------------------------------------------------------ #
# 1. Loss curve
try:
    plt.figure()
    plt.plot(epochs, loss_tr, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{ds_name}: {model_name} Loss Curves")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_{model_name}_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2. CWA curve
try:
    plt.figure()
    plt.plot(epochs, cwa_tr, label="Train")
    plt.plot(epochs, cwa_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Color-Weighted Accuracy")
    plt.title(f"{ds_name}: {model_name} CWA Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_{model_name}_CWA.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CWA plot: {e}")
    plt.close()

# 3. SWA curve
try:
    plt.figure()
    plt.plot(epochs, swa_tr, label="Train")
    plt.plot(epochs, swa_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title(f"{ds_name}: {model_name} SWA Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_{model_name}_SWA.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA plot: {e}")
    plt.close()

# 4. Complexity-Weighted Accuracy curve
try:
    plt.figure()
    plt.plot(epochs, cpx_tr, label="Train")
    plt.plot(epochs, cpx_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Complexity-Weighted Accuracy")
    plt.title(f"{ds_name}: {model_name} CmpWA Curves")
    plt.legend()
    plt.savefig(os.path.join(working_dir, f"{ds_name}_{model_name}_CmpWA.png"))
    plt.close()
except Exception as e:
    print(f"Error creating CmpWA plot: {e}")
    plt.close()

# 5. Confusion matrix heat-map (max 5 plots total)
try:
    import itertools

    num_cls = max(max(y_true, default=0), max(y_pred, default=0)) + 1
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{ds_name}: {model_name} Confusion Matrix")
    for i, j in itertools.product(range(num_cls), range(num_cls)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            fontsize=8,
        )
    plt.savefig(
        os.path.join(working_dir, f"{ds_name}_{model_name}_confusion_matrix.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------------------------------------------------------ #
# print test metrics
test_metrics = exp.get("test_metrics", {})
print("Test Metrics:")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
