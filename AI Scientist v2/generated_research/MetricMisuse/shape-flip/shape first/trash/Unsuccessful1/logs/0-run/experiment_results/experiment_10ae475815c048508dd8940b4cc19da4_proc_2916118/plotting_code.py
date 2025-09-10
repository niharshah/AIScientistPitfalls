import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------- load data ----------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "SPR_BENCH"
data = experiment_data.get(ds_key, {})

# --------------------------- figure 1 ------------------------------
try:
    plt.figure()
    train_loss = data["losses"]["train"]
    val_loss = data["losses"]["val"]
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{ds_key}: Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, f"{ds_key.lower()}_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# --------------------------- figure 2 ------------------------------
try:
    plt.figure()
    hwa_vals = [m["hwa"] for m in data["metrics"]["val"]]
    plt.plot(range(1, len(hwa_vals) + 1), hwa_vals)
    plt.xlabel("Epoch")
    plt.ylabel("HWA")
    plt.title(f"{ds_key}: Validation HWA over Epochs")
    fname = os.path.join(working_dir, f"{ds_key.lower()}_val_hwa.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating HWA curve: {e}")
    plt.close()

# --------------------------- figure 3 ------------------------------
try:
    plt.figure()
    test_metrics = data["metrics"]["test"]
    names = ["swa", "cwa", "hwa", "loss"]
    values = [test_metrics.get(k, np.nan) for k in names]
    plt.bar(names, values, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    plt.title(f"{ds_key}: Final Test Metrics")
    plt.ylabel("Value")
    fname = os.path.join(working_dir, f"{ds_key.lower()}_test_metrics.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar: {e}")
    plt.close()

# --------------------------- figure 4 ------------------------------
try:
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])
    if preds.size and gts.size:
        labels = sorted(set(np.concatenate([preds, gts])))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for p, g in zip(preds, gts):
            cm[label_to_idx[g], label_to_idx[p]] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"{ds_key}: Confusion Matrix (Test Set)")
        fname = os.path.join(working_dir, f"{ds_key.lower()}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
