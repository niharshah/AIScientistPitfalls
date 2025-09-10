import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ LOAD EXPERIMENT DATA ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

saved_figs = []

# ------------------ ITERATE AND PLOT -------------------------------
for dname, ddict in experiment_data.items():
    losses = ddict.get("losses", {})
    metrics = ddict.get("metrics", {})
    irf = ddict.get("IRF", {})
    y_pred = np.array(ddict.get("predictions", []))
    y_true = np.array(ddict.get("ground_truth", []))

    # 1) Train & Validation loss
    try:
        tr_loss, val_loss = losses.get("train", []), losses.get("val", [])
        if tr_loss and val_loss:
            plt.figure()
            epochs = range(1, len(tr_loss) + 1)
            plt.plot(epochs, tr_loss, marker="o", label="Train")
            plt.plot(epochs, val_loss, marker="s", label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} – Train vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_train_val_loss.png")
            plt.savefig(fname)
            saved_figs.append(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # 2) Train & Validation accuracy
    try:
        tr_acc, val_acc = metrics.get("train", []), metrics.get("val", [])
        if tr_acc and val_acc:
            plt.figure()
            epochs = range(1, len(tr_acc) + 1)
            plt.plot(epochs, tr_acc, marker="o", label="Train", color="purple")
            plt.plot(epochs, val_acc, marker="s", label="Validation", color="green")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname} – Train vs Validation Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_train_val_accuracy.png")
            plt.savefig(fname)
            saved_figs.append(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dname}: {e}")
        plt.close()

    # 3) IRF bar chart
    try:
        irf_val = irf.get("val", [])
        irf_test = irf.get("test", [])
        if irf_val and irf_test:
            plt.figure()
            plt.bar(
                ["Dev", "Test"],
                [irf_val[-1], irf_test[-1]],
                color=["skyblue", "orange"],
            )
            plt.ylim(0, 1)
            plt.ylabel("IRF")
            plt.title(f"{dname} – Interpretable Representation Fidelity")
            fname = os.path.join(working_dir, f"{dname}_irf.png")
            plt.savefig(fname)
            saved_figs.append(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating IRF plot for {dname}: {e}")
        plt.close()

    # 4) Confusion matrix
    try:
        if y_true.size and y_pred.size:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.title(f"{dname} – Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center")
            plt.colorbar(im)
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname)
            saved_figs.append(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

    # 5) Class distribution
    try:
        if y_true.size and y_pred.size:
            classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            gt_counts = [np.sum(y_true == c) for c in classes]
            pred_counts = [np.sum(y_pred == c) for c in classes]
            x = np.arange(len(classes))
            width = 0.35
            plt.figure()
            plt.bar(x - width / 2, gt_counts, width, label="Ground Truth")
            plt.bar(x + width / 2, pred_counts, width, label="Predicted")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(f"{dname} – Class Distribution")
            plt.xticks(x, classes)
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_class_distribution.png")
            plt.savefig(fname)
            saved_figs.append(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating class distribution plot for {dname}: {e}")
        plt.close()

print("Saved figures:")
for f in saved_figs:
    print(" -", os.path.basename(f))
