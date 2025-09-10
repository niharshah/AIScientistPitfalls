import matplotlib.pyplot as plt
import numpy as np
import os

# ----- paths -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


def safe(arr, key):
    return np.array(arr.get(key, []))


final_val_accs = {}

for dset, blob in experiment_data.items():
    metrics, losses = blob.get("metrics", {}), blob.get("losses", {})
    # Plot 1: Accuracy curves
    try:
        tr_acc, vl_acc = safe(metrics, "train_acc"), safe(metrics, "val_acc")
        if tr_acc.size and vl_acc.size:
            epochs = np.arange(1, len(tr_acc) + 1)
            plt.figure()
            plt.plot(epochs, tr_acc, label="Train Acc")
            plt.plot(epochs, vl_acc, label="Val Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dset}: Train vs Validation Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dset}_accuracy_curves.png"))
            plt.close()
            final_val_accs[dset] = vl_acc[-1]
    except Exception as e:
        print(f"{dset}: accuracy plot error: {e}")
        plt.close()

    # Plot 2: Loss curves
    try:
        tr_ls, vl_ls = safe(losses, "train"), safe(losses, "val")
        if tr_ls.size and vl_ls.size:
            epochs = np.arange(1, len(tr_ls) + 1)
            plt.figure()
            plt.plot(epochs, tr_ls, label="Train Loss")
            plt.plot(epochs, vl_ls, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("CE Loss")
            plt.title(f"{dset}: Train vs Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dset}_loss_curves.png"))
            plt.close()
    except Exception as e:
        print(f"{dset}: loss plot error: {e}")
        plt.close()

    # Plot 3: Rule Fidelity vs Val Acc
    try:
        fid = safe(metrics, "Rule_Fidelity")
        vl_acc = safe(metrics, "val_acc")
        if fid.size and vl_acc.size:
            epochs = np.arange(1, len(fid) + 1)
            plt.figure()
            plt.plot(epochs, vl_acc, label="Val Acc")
            plt.plot(epochs, fid, label="Rule Fidelity")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dset}: Validation Accuracy vs Rule Fidelity")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dset}_val_vs_fidelity.png"))
            plt.close()
    except Exception as e:
        print(f"{dset}: fidelity plot error: {e}")
        plt.close()

    # Plot 4: Confusion matrix
    try:
        preds, gts = safe(blob, "predictions"), safe(blob, "ground_truth")
        if preds.size and gts.size and preds.shape == gts.shape:
            n_cls = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((n_cls, n_cls), int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dset}: Confusion Matrix (Test Set)")
            plt.savefig(os.path.join(working_dir, f"{dset}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"{dset}: confusion plot error: {e}")
        plt.close()

    # Print test accuracy if available
    if preds.size and gts.size and preds.shape == gts.shape:
        print(f"{dset} Test Accuracy: {(preds==gts).mean():.3f}")

# Cross-dataset comparison bar chart
try:
    if final_val_accs:
        plt.figure()
        names = list(final_val_accs.keys())
        vals = [final_val_accs[k] for k in names]
        plt.bar(names, vals)
        plt.ylabel("Final Validation Accuracy")
        plt.title("Comparison of Final Validation Accuracy Across Datasets")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "comparison_final_val_accuracy.png"))
        plt.close()
except Exception as e:
    print(f"Comparison plot error: {e}")
    plt.close()
