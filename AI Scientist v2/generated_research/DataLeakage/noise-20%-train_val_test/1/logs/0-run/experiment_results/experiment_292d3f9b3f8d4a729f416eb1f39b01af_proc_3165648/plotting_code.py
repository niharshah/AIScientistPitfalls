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

# iterate through all stored model/dataset results
for model_name, datasets in experiment_data.items():
    for dset_name, d in datasets.items():
        # ------------------------- curves ----------------------------- #
        metrics, losses = d.get("metrics", {}), d.get("losses", {})
        train_loss, val_loss = map(
            np.array, (losses.get("train", []), losses.get("val", []))
        )
        train_acc, val_acc = map(
            np.array, (metrics.get("train_acc", []), metrics.get("val_acc", []))
        )
        train_f1, val_f1 = map(
            np.array, (metrics.get("train_f1", []), metrics.get("val_f1", []))
        )
        epochs = np.arange(1, max(len(train_loss), len(train_acc), len(train_f1)) + 1)

        # 1. loss curves
        if train_loss.size and val_loss.size:
            try:
                plt.figure()
                plt.plot(epochs[: len(train_loss)], train_loss, label="Train Loss")
                plt.plot(epochs[: len(val_loss)], val_loss, label="Val Loss")
                plt.title(f"{dset_name} Loss Curves (Model: {model_name})")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(
                    os.path.join(
                        working_dir, f"{model_name}_{dset_name}_loss_curves.png"
                    )
                )
                plt.close()
            except Exception as e:
                print(f"Error creating loss curve for {dset_name}: {e}")
                plt.close()

        # 2. accuracy curves
        if train_acc.size and val_acc.size:
            try:
                plt.figure()
                plt.plot(epochs[: len(train_acc)], train_acc, label="Train Acc")
                plt.plot(epochs[: len(val_acc)], val_acc, label="Val Acc")
                plt.title(f"{dset_name} Accuracy Curves (Model: {model_name})")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.savefig(
                    os.path.join(
                        working_dir, f"{model_name}_{dset_name}_accuracy_curves.png"
                    )
                )
                plt.close()
            except Exception as e:
                print(f"Error creating accuracy curve for {dset_name}: {e}")
                plt.close()

        # 3. macro-F1 curves
        if train_f1.size and val_f1.size:
            try:
                plt.figure()
                plt.plot(epochs[: len(train_f1)], train_f1, label="Train F1")
                plt.plot(epochs[: len(val_f1)], val_f1, label="Val F1")
                plt.title(f"{dset_name} Macro-F1 Curves (Model: {model_name})")
                plt.xlabel("Epoch")
                plt.ylabel("Macro-F1")
                plt.legend()
                plt.savefig(
                    os.path.join(working_dir, f"{model_name}_{dset_name}_f1_curves.png")
                )
                plt.close()
            except Exception as e:
                print(f"Error creating F1 curve for {dset_name}: {e}")
                plt.close()

        # 4. confusion matrix (test set)
        preds, gts = map(
            np.asarray, (d.get("predictions", []), d.get("ground_truth", []))
        )
        if preds.size and gts.size:
            try:
                n_cls = int(max(preds.max(), gts.max()) + 1)
                cm = np.zeros((n_cls, n_cls), dtype=int)
                for gt, pr in zip(gts, preds):
                    cm[gt, pr] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.title(f"{dset_name} Confusion Matrix (Model: {model_name})")
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                ticks = np.arange(n_cls)
                plt.xticks(ticks)
                plt.yticks(ticks)
                plt.savefig(
                    os.path.join(
                        working_dir, f"{model_name}_{dset_name}_confusion_matrix.png"
                    )
                )
                plt.close()
                # print evaluation metrics for quick reference
                test_acc = (preds == gts).mean()
                test_f1 = d.get("test_f1", None)
                print(
                    f"{model_name}/{dset_name} -- Test Acc: {test_acc*100:.2f}% | Test Macro-F1: {test_f1:.4f}"
                    if test_f1 is not None
                    else f"{model_name}/{dset_name} -- Test Acc: {test_acc*100:.2f}%"
                )
            except Exception as e:
                print(f"Error creating confusion matrix for {dset_name}: {e}")
                plt.close()
