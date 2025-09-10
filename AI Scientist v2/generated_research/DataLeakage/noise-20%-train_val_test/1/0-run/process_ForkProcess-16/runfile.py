import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def macro_f1(preds, labels, num_cls):
    f1s = []
    for c in range(num_cls):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


# -------------------- load experiment data --------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Only proceed if data exists
for exp_name, ds_dict in experiment_data.items():
    for ds_name, record in ds_dict.items():
        losses = record.get("losses", {})
        metrics = record.get("metrics", {})
        preds = np.asarray(record.get("predictions", []))
        gts = np.asarray(record.get("ground_truth", []))
        epochs = range(1, 1 + len(losses.get("train", [])))

        # --------------- plot 1: loss curves -------------------
        try:
            if losses:
                plt.figure()
                plt.plot(epochs, losses["train"], label="Train")
                plt.plot(epochs, losses["val"], label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{ds_name}: Training vs Validation Loss")
                plt.legend()
                fname = f"{ds_name}_loss_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating loss plot: {e}")
            plt.close()

        # --------------- plot 2: accuracy & F1 -----------------
        try:
            if metrics:
                train_acc = [m["acc"] for m in metrics["train"]]
                val_acc = [m["acc"] for m in metrics["val"]]
                train_f1 = [m["f1"] for m in metrics["train"]]
                val_f1 = [m["f1"] for m in metrics["val"]]

                fig, ax1 = plt.subplots()
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Accuracy")
                ax1.plot(epochs, train_acc, "b-", label="Train Acc")
                ax1.plot(epochs, val_acc, "b--", label="Val Acc")
                ax2 = ax1.twinx()
                ax2.set_ylabel("Macro-F1")
                ax2.plot(epochs, train_f1, "r-", label="Train F1")
                ax2.plot(epochs, val_f1, "r--", label="Val F1")

                lines, labs = ax1.get_legend_handles_labels()
                lines2, labs2 = ax2.get_legend_handles_labels()
                ax1.legend(lines + lines2, labs + labs2, loc="lower center")
                plt.title(f"{ds_name}: Accuracy & Macro-F1")
                plt.tight_layout()
                fname = f"{ds_name}_metric_curves.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating metric plot: {e}")
            plt.close()

        # --------------- plot 3: confusion matrix --------------
        try:
            if preds.size and gts.size:
                num_cls = int(max(gts.max(), preds.max()) + 1)
                cm = np.zeros((num_cls, num_cls), int)
                for p, t in zip(preds, gts):
                    cm[t, p] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"{ds_name}: Confusion Matrix")
                plt.tight_layout()
                fname = f"{ds_name}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()

                acc = (preds == gts).mean()
                f1 = macro_f1(preds, gts, num_cls)
                print(f"{ds_name} Test Accuracy: {acc*100:.2f}% | Macro-F1: {f1:.4f}")
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()
