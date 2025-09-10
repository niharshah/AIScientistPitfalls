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


# helper
def safe(arr, key):
    return np.array(arr.get(key, []))


all_val_acc = {}
# -------- per-dataset plots -------- #
for dname, data in experiment_data.items():
    metrics, losses = data.get("metrics", {}), data.get("losses", {})
    train_acc, val_acc = safe(metrics, "train_acc"), safe(metrics, "val_acc")
    train_loss, val_loss = safe(losses, "train"), safe(losses, "val")
    rule_fid = safe(metrics, "Rule_Fidelity")
    preds, gts = np.array(data.get("predictions", [])), np.array(
        data.get("ground_truth", [])
    )
    all_val_acc[dname] = val_acc

    # Plot 1: accuracy
    try:
        if train_acc.size and val_acc.size:
            plt.figure()
            ep = np.arange(1, len(train_acc) + 1)
            plt.plot(ep, train_acc, label="Train Acc")
            plt.plot(ep, val_acc, label="Val Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname}: Train vs Validation Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_accuracy_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error accuracy plot for {dname}: {e}")
        plt.close()

    # Plot 2: loss
    try:
        if train_loss.size and val_loss.size:
            plt.figure()
            ep = np.arange(1, len(train_loss) + 1)
            plt.plot(ep, train_loss, label="Train Loss")
            plt.plot(ep, val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname}: Train vs Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error loss plot for {dname}: {e}")
        plt.close()

    # Plot 3: rule fidelity vs val acc
    try:
        if rule_fid.size and val_acc.size:
            plt.figure()
            ep = np.arange(1, len(rule_fid) + 1)
            plt.plot(ep, val_acc, label="Val Acc")
            plt.plot(ep, rule_fid, label="Rule Fidelity")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname}: Validation Accuracy vs Rule Fidelity")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_val_vs_rulefid.png"))
            plt.close()
    except Exception as e:
        print(f"Error rule fidelity plot for {dname}: {e}")
        plt.close()

    # Plot 4: confusion matrix
    try:
        if preds.size and gts.size and preds.shape == gts.shape:
            n_cls = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dname}: Confusion Matrix (Test)")
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error confusion matrix for {dname}: {e}")
        plt.close()

    # Print test accuracy
    if preds.size and gts.size and preds.shape == gts.shape:
        print(f"{dname} Test Accuracy: {(preds==gts).mean():.3f}")

# -------- comparison plot across datasets -------- #
try:
    if len(all_val_acc) >= 2:
        plt.figure()
        for dname, v in all_val_acc.items():
            if v.size:
                plt.plot(np.arange(1, len(v) + 1), v, label=f"{dname} Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Dataset Comparison: Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_val_accuracy.png"))
        plt.close()
except Exception as e:
    print(f"Error comparison plot: {e}")
    plt.close()
