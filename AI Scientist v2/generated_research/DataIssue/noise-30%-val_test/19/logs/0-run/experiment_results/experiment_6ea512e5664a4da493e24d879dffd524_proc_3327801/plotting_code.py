import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- Load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dname, d in experiment_data.items():
    epochs = d.get("epochs", [])
    tr_loss = d["losses"].get("train", [])
    val_loss = d["losses"].get("val", [])
    tr_f1 = d["metrics"].get("train_f1", [])
    val_f1 = d["metrics"].get("val_f1", [])
    preds = np.array(d.get("predictions", []))
    golds = np.array(d.get("ground_truth", []))
    n_classes = len(np.unique(np.concatenate([preds, golds]))) if len(preds) else 0

    # 1) Loss curve
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname} – Loss vs Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) F1 curve
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{dname} – Macro-F1 vs Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dname}_f1_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # 3) Confusion matrix
    try:
        if len(preds):
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for g, p in zip(golds, preds):
                cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dname} – Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # 4) Class distribution bar chart
    try:
        if len(preds):
            labels = np.arange(n_classes)
            width = 0.35
            counts_gold = np.bincount(golds, minlength=n_classes)
            counts_pred = np.bincount(preds, minlength=n_classes)
            plt.figure()
            plt.bar(labels - width / 2, counts_gold, width, label="Ground Truth")
            plt.bar(labels + width / 2, counts_pred, width, label="Predictions")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(f"{dname} – Class Distribution")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dname}_class_distribution.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating class distribution plot: {e}")
        plt.close()

    # Print evaluation metric
    print(f"{dname} test_macro_f1: {d.get('test_f1', 'N/A')}")
