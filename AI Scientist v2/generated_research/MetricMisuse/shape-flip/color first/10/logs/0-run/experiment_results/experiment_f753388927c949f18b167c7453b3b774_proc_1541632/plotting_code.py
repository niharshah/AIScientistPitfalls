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

for dname, ddata in experiment_data.items():
    # ----- gather losses -----
    train_losses = np.asarray(ddata["losses"].get("train", []), dtype=float)
    val_losses = np.asarray(ddata["losses"].get("val", []), dtype=float)
    epochs = np.arange(1, len(train_losses) + 1)

    # ----- gather validation metrics -----
    val_metrics = ddata["metrics"].get("val", [])
    cwa = [m["cwa"] for m in val_metrics] if val_metrics else []
    swa = [m["swa"] for m in val_metrics] if val_metrics else []
    hwa = [m["hwa"] for m in val_metrics] if val_metrics else []

    # ----- build test confusion matrix -----
    preds = ddata.get("predictions", [])
    gts = ddata.get("ground_truth", [])
    labels = sorted(set(gts + preds))
    lab2idx = {l: i for i, l in enumerate(labels)}
    conf = np.zeros((len(labels), len(labels)), dtype=int)
    for gt, pr in zip(gts, preds):
        conf[lab2idx[gt], lab2idx[pr]] += 1

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_losses, label="Train")
        plt.plot(epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname} Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # 2) Metric curves
    try:
        plt.figure()
        if cwa:
            plt.plot(epochs, cwa, label="CWA")
        if swa:
            plt.plot(epochs, swa, label="SWA")
        if hwa:
            plt.plot(epochs, hwa, label="HWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dname} Validation Weighted Accuracies")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_metric_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {dname}: {e}")
        plt.close()

    # 3) Confusion matrix
    try:
        if conf.size:
            plt.figure()
            im = plt.imshow(conf, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dname} Confusion Matrix (Test)")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, str(conf[i, j]), ha="center", va="center")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

    # ----- print test metrics -----
    test_metrics = ddata["metrics"].get("test", {})
    if test_metrics:
        print(
            f"{dname} Test Metrics: "
            + ", ".join(f"{k.upper()}={v:.3f}" for k, v in test_metrics.items())
        )
