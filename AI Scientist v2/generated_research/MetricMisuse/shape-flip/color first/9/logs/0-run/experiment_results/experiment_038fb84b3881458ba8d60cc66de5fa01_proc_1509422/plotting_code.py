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

for ds_name, ds in experiment_data.items():
    # 1) Loss curves ----------------------------------------------------------
    try:
        epochs = [d["epoch"] for d in ds["losses"]["train"]]
        train_loss = [d["loss"] for d in ds["losses"]["train"]]
        val_loss = [d["loss"] for d in ds["losses"]["val"]]
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.title(f"{ds_name}: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) Metric curves --------------------------------------------------------
    try:
        mets = ds["metrics"]["val"]
        epochs = [m["epoch"] for m in mets]
        sdwa = [m["sdwa"] for m in mets]
        cwa = [m["cwa"] for m in mets]
        swa = [m["swa"] for m in mets]
        plt.figure()
        plt.plot(epochs, sdwa, label="SDWA")
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{ds_name}: Validation Metrics over Epochs")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_metric_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # 3) Confusion matrix -----------------------------------------------------
    try:
        y_true = np.array(ds["ground_truth"])
        y_pred = np.array(ds["predictions"])
        labels = sorted(set(np.concatenate([y_true, y_pred])))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[labels.index(t), labels.index(p)] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{ds_name}: Confusion Matrix")
        plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix.png"))
        plt.close()
        # quick numeric summary
        acc = (y_true == y_pred).mean() if len(y_true) else 0.0
        sdwa_test = ds["metrics"]["val"][-1]["sdwa"] if ds["metrics"]["val"] else 0.0
        print(
            f"{ds_name} – Test Accuracy: {acc:.4f} | Last-epoch SDWA (val): {sdwa_test:.4f}"
        )
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
