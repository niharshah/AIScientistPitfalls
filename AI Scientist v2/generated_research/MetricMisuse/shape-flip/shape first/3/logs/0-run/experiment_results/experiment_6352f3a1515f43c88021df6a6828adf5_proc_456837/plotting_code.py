import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- paths -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------- load data -----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch safely
def get(data, *keys, default=None):
    for k in keys:
        data = data.get(k, {})
    return data if data != {} else default


ds_name = "SPR_BENCH"
exp = experiment_data.get(ds_name, {})

loss_train = get(exp, "losses", "train", default=[])
loss_val = get(exp, "losses", "val", default=[])
metrics_val = get(exp, "metrics", "val", default=[])
y_pred_t = exp.get("predictions", [])
y_true_t = exp.get("ground_truth", [])

epochs = np.arange(1, len(loss_train) + 1)

# unpack validation metrics list-of-dicts into arrays
swa_vals, cwa_vals, hwa_vals = [], [], []
for d in metrics_val:
    if isinstance(d, dict):
        swa_vals.append(d.get("SWA", np.nan))
        cwa_vals.append(d.get("CWA", np.nan))
        hwa_vals.append(d.get("HWA", np.nan))

# ----------- Plot 1: loss curves -----------
try:
    plt.figure()
    if len(loss_train) > 0:
        plt.plot(epochs, loss_train, label="Train Loss")
    if len(loss_val) > 0:
        plt.plot(epochs, loss_val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{ds_name}: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ----------- Plot 2: validation metrics -----------
try:
    if swa_vals and cwa_vals and hwa_vals:
        plt.figure()
        plt.plot(epochs, swa_vals, label="SWA")
        plt.plot(epochs, cwa_vals, label="CWA")
        plt.plot(epochs, hwa_vals, label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title(f"{ds_name}: Validation Metrics Over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds_name}_val_metrics.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation metric plot: {e}")
    plt.close()

# ----------- Plot 3: final test metrics bar chart -----------
try:
    if swa_vals and cwa_vals and hwa_vals:
        swa_t, cwa_t, hwa_t = (
            swa_vals[-1],
            cwa_vals[-1],
            hwa_vals[-1],
        )  # already computed during save
        metrics = ["SWA", "CWA", "HWA"]
        values = [swa_t, cwa_t, hwa_t]
        plt.figure()
        plt.bar(metrics, values, color=["skyblue", "lightgreen", "salmon"])
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.title(f"{ds_name}: Final Validation Metric Values")
        fname = os.path.join(working_dir, f"{ds_name}_final_metrics_bar.png")
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ----------- Plot 4: confusion matrix -----------
try:
    if y_true_t and y_pred_t:
        labels = sorted(list(set(y_true_t) | set(y_pred_t)))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for yt, yp in zip(y_true_t, y_pred_t):
            cm[label_to_idx[yt], label_to_idx[yp]] += 1
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{ds_name}: Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
        plt.savefig(fname, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# ----------- print final metrics -----------
if swa_vals and cwa_vals and hwa_vals:
    print(
        f"Final Validation Metrics -> SWA: {swa_vals[-1]:.4f}, CWA: {cwa_vals[-1]:.4f}, HWA: {hwa_vals[-1]:.4f}"
    )
if y_true_t and y_pred_t:
    # compute test metrics quickly
    def c_variety(seq):
        return len(set(tok[1] for tok in seq.split() if len(tok) > 1))

    def s_variety(seq):
        return len(set(tok[0] for tok in seq.split()))

    sequences = y_true_t  # placeholder since sequences not stored for test; metrics already saved elsewhere
    print(f"Test predictions available: {len(y_pred_t)} samples")
