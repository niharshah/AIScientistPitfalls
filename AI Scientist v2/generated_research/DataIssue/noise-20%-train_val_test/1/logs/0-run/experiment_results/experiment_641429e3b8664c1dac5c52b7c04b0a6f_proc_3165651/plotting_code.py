import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------- #
# load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# --------------------------------------------------------------------- #
# iterate over model / dataset pairs
for model_name, model_blob in experiment_data.items():
    for dataset_name, data in model_blob.items():

        # fetch arrays safely
        train_loss = np.asarray(data["losses"].get("train", []), dtype=float)
        val_loss = np.asarray(data["losses"].get("val", []), dtype=float)
        train_f1 = np.asarray(data["metrics"].get("train_f1", []), dtype=float)
        val_f1 = np.asarray(data["metrics"].get("val_f1", []), dtype=float)
        preds = np.asarray(data.get("predictions", []), dtype=int)
        gts = np.asarray(data.get("ground_truth", []), dtype=int)
        epochs = np.arange(1, len(train_loss) + 1)

        tag = f"{dataset_name.lower()}_{model_name.lower()}"

        # ------------------------------------------------------------- #
        # 1. Loss curves
        try:
            if train_loss.size and val_loss.size:
                plt.figure()
                plt.plot(epochs, train_loss, label="Train")
                plt.plot(epochs, val_loss, label="Validation")
                plt.title(f"{dataset_name} ({model_name}) Loss Curves")
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.legend()
                plt.savefig(os.path.join(working_dir, f"{tag}_loss_curves.png"))
                plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {tag}: {e}")
            plt.close()

        # ------------------------------------------------------------- #
        # 2. F1 curves
        try:
            if train_f1.size and val_f1.size:
                plt.figure()
                plt.plot(epochs, train_f1, label="Train")
                plt.plot(epochs, val_f1, label="Validation")
                plt.title(f"{dataset_name} ({model_name}) Macro-F1 Curves")
                plt.xlabel("Epoch")
                plt.ylabel("Macro-F1")
                plt.legend()
                plt.savefig(os.path.join(working_dir, f"{tag}_f1_curves.png"))
                plt.close()
        except Exception as e:
            print(f"Error creating F1 curve for {tag}: {e}")
            plt.close()

        # ------------------------------------------------------------- #
        # 3. Confusion matrix on test set
        try:
            if preds.size and gts.size:
                num_classes = int(max(preds.max(), gts.max()) + 1)
                cm = np.zeros((num_classes, num_classes), dtype=int)
                for y_true, y_pred in zip(gts, preds):
                    cm[y_true, y_pred] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.title(f"{dataset_name} ({model_name}) Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                ticks = np.arange(num_classes)
                plt.xticks(ticks)
                plt.yticks(ticks)
                plt.savefig(os.path.join(working_dir, f"{tag}_confusion_matrix.png"))
                plt.close()

                test_acc = (preds == gts).mean()
                # simple Macro-F1 for confirmation
                f1_per_class = []
                for c in range(num_classes):
                    tp = np.sum((gts == c) & (preds == c))
                    fp = np.sum((gts != c) & (preds == c))
                    fn = np.sum((gts == c) & (preds != c))
                    prec = tp / (tp + fp + 1e-9) if tp + fp else 0.0
                    rec = tp / (tp + fn + 1e-9) if tp + fn else 0.0
                    f1 = 2 * prec * rec / (prec + rec + 1e-9) if prec + rec else 0.0
                    f1_per_class.append(f1)
                print(
                    f"{tag}: Test Acc={test_acc*100:.2f}%  Macro-F1={np.mean(f1_per_class)*100:.2f}%"
                )
        except Exception as e:
            print(f"Error creating confusion matrix for {tag}: {e}")
            plt.close()
