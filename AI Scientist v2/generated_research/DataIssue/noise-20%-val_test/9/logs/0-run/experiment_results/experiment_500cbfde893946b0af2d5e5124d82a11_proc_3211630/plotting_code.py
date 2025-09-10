import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- paths & load ----------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


def safe_get(dct, *keys):
    cur = dct
    for k in keys:
        cur = cur.get(k, {})
    return np.array(cur)


# ---------------- iterate over datasets ----------------
for ds_name, ds_blob in experiment_data.items():
    metrics = ds_blob.get("metrics", {})
    losses = ds_blob.get("losses", {})
    preds = np.array(ds_blob.get("predictions", []))
    gts = np.array(ds_blob.get("ground_truth", []))

    # -- Plot 1: Accuracy curves --
    try:
        train_acc = safe_get(metrics, "train_acc")
        val_acc = safe_get(metrics, "val_acc")
        if train_acc.size and val_acc.size:
            plt.figure()
            epochs = np.arange(1, len(train_acc) + 1)
            plt.plot(epochs, train_acc, label="Train")
            plt.plot(epochs, val_acc, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{ds_name}: Train vs Validation Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_accuracy_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting accuracy for {ds_name}: {e}")
        plt.close()

    # -- Plot 2: Loss curves --
    try:
        train_loss = safe_get(losses, "train")
        val_loss = safe_get(losses, "val")
        if train_loss.size and val_loss.size:
            plt.figure()
            epochs = np.arange(1, len(train_loss) + 1)
            plt.plot(epochs, train_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{ds_name}: Train vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting loss for {ds_name}: {e}")
        plt.close()

    # -- Plot 3: Rule Fidelity curves --
    try:
        rf = safe_get(metrics, "Rule_Fidelity")
        if rf.size:
            plt.figure()
            epochs = np.arange(1, len(rf) + 1)
            plt.plot(epochs, rf, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Rule Fidelity")
            plt.title(f"{ds_name}: Rule Fidelity over Epochs")
            fname = os.path.join(working_dir, f"{ds_name}_rule_fidelity.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting rule fidelity for {ds_name}: {e}")
        plt.close()

    # -- Plot 4: Confusion matrix (Test) --
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
            plt.title(f"{ds_name}: Confusion Matrix (Test)")
            fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {ds_name}: {e}")
        plt.close()

    # -------- evaluation metric printout --------
    if preds.size and gts.size and preds.shape == gts.shape:
        acc = (preds == gts).mean()
        print(f"{ds_name} Test Accuracy: {acc:.3f}")
