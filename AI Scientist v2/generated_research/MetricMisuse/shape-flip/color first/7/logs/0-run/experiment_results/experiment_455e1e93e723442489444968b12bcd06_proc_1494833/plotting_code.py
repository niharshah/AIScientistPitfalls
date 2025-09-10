import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helpers ----------
def comp_weight(seq):
    toks = seq.split()
    color_var = len(set(t[1:] for t in toks if len(t) > 1))
    shape_var = len(set(t[0] for t in toks if t))
    return color_var + shape_var


# ---------- per-dataset plots ----------
val_acc_for_cmp = {}
for ds_name, ds in experiment_data.items():
    epochs = np.arange(1, len(ds["losses"]["train"]) + 1)
    train_loss = np.array(ds["losses"]["train"])
    val_loss = np.array(ds["losses"]["val"])
    train_acc = np.array([m["acc"] for m in ds["metrics"]["train"]])
    val_acc = np.array([m["acc"] for m in ds["metrics"]["val"]])
    compwa = np.array([m.get("CompWA", m.get("cowa", 0)) for m in ds["metrics"]["val"]])
    val_acc_for_cmp[ds_name] = (epochs, val_acc)

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} Accuracy Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()

    # 3) CompWA curves
    try:
        plt.figure()
        plt.plot(epochs, compwa, label="Validation CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        plt.title(f"{ds_name} Complexity-Weighted Accuracy Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_compwa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot for {ds_name}: {e}")
        plt.close()

    # 4) Histogram of complexity weight (correct vs incorrect)
    try:
        preds = np.array(ds["predictions"])
        gts = np.array(ds["ground_truth"])
        seqs = np.array(ds["sequences"])
        weights = np.array([comp_weight(s) for s in seqs])
        correct_w = weights[preds == gts]
        wrong_w = weights[preds != gts]
        plt.figure()
        bins = np.arange(weights.min(), weights.max() + 2) - 0.5
        plt.hist(correct_w, bins=bins, alpha=0.7, label="Correct", edgecolor="black")
        plt.hist(wrong_w, bins=bins, alpha=0.7, label="Incorrect", edgecolor="black")
        plt.xlabel("Complexity Weight")
        plt.ylabel("Count")
        plt.title(f"{ds_name} Prediction Outcome vs. Complexity")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_complexity_hist.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating complexity histogram for {ds_name}: {e}")
        plt.close()

    # ---------- evaluation metrics ----------
    try:
        test_acc = (preds == gts).mean()
        cowa = (weights * (preds == gts)).sum() / weights.sum()
        print(f"{ds_name} -- Test Accuracy: {test_acc:.3f} | Test CompWA: {cowa:.3f}")
    except Exception as e:
        print(f"Error computing metrics for {ds_name}: {e}")

# ---------- comparison plot (if >1 dataset) ----------
if len(val_acc_for_cmp) > 1:
    try:
        plt.figure()
        for name, (ep, acc) in val_acc_for_cmp.items():
            plt.plot(ep, acc, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Dataset Comparison: Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        plt.close()
