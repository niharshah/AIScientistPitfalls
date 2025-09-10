import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
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


# ---------- helper ----------
def safe(arr, key):
    return np.array(arr.get(key, []))


# ---------- per-dataset plots ----------
all_datasets = list(experiment_data.keys())
val_acc_dict = {}

for dname, ddata in experiment_data.items():
    metrics, losses = ddata.get("metrics", {}), ddata.get("losses", {})
    train_acc, val_acc = safe(metrics, "train_acc"), safe(metrics, "val_acc")
    train_loss, val_loss = safe(losses, "train"), safe(losses, "val")
    rule_fid = safe(metrics, "Rule_Fidelity")
    preds, gts = np.array(ddata.get("predictions", [])), np.array(
        ddata.get("ground_truth", [])
    )

    # 1. Accuracy curves
    try:
        if train_acc.size and val_acc.size:
            epochs = np.arange(1, len(train_acc) + 1)
            plt.figure()
            plt.plot(epochs, train_acc, label="Train Acc")
            plt.plot(epochs, val_acc, label="Val Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname}: Train vs Validation Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_accuracy_curves.png"))
            plt.close()
    except Exception as e:
        print(f"{dname}: error accuracy plot {e}")
        plt.close()

    # 2. Loss curves
    try:
        if train_loss.size and val_loss.size:
            epochs = np.arange(1, len(train_loss) + 1)
            plt.figure()
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname}: Train vs Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
            plt.close()
    except Exception as e:
        print(f"{dname}: error loss plot {e}")
        plt.close()

    # 3. Rule Fidelity vs Val Acc
    try:
        if rule_fid.size and val_acc.size:
            epochs = np.arange(1, len(rule_fid) + 1)
            plt.figure()
            plt.plot(epochs, val_acc, label="Val Acc")
            plt.plot(epochs, rule_fid, label="Rule Fidelity")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy / Fidelity")
            plt.title(f"{dname}: Validation Accuracy vs Rule Fidelity")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_rulefid_vs_val.png"))
            plt.close()
    except Exception as e:
        print(f"{dname}: error rule fidelity plot {e}")
        plt.close()

    # 4. Confusion Matrix
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
        print(f"{dname}: error confusion matrix {e}")
        plt.close()

    # Store val_acc for cross-dataset comparison
    if val_acc.size:
        val_acc_dict[dname] = val_acc

    # Print evaluation metric
    if preds.size and preds.shape == gts.shape:
        print(f"{dname} Test Accuracy: {(preds == gts).mean():.3f}")

# ---------- Cross-dataset comparison plot ----------
try:
    if len(val_acc_dict) > 1:
        plt.figure()
        for dname, v in val_acc_dict.items():
            epochs = np.arange(1, len(v) + 1)
            plt.plot(epochs, v, label=f"{dname}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Dataset Comparison: Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_val_accuracy.png"))
        plt.close()
except Exception as e:
    print(f"Error creating cross-dataset comparison: {e}")
    plt.close()
