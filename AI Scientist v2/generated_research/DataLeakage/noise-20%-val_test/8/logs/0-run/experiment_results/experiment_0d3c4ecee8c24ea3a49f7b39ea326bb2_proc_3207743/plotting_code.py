import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ SETUP ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ LOAD DATA -----------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------ PLOT & PRINT --------------------
for dname, ddict in experiment_data.items():
    losses = ddict.get("losses", {})
    metrics = ddict.get("metrics", {})
    y_pred = np.array(ddict.get("predictions", []))
    y_true = np.array(ddict.get("ground_truth", []))
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    train_acc = metrics.get("train_acc", [])
    val_acc = metrics.get("val_acc", [])
    irf_list = metrics.get("IRF", [])
    irf_val = irf_list[0] if len(irf_list) else None

    # ---------------- PRINT METRICS -----------------
    if len(val_acc):
        print(f"{dname}: final val acc = {val_acc[-1]:.4f}")
    if y_true.size and y_pred.size:
        from sklearn.metrics import accuracy_score

        print(f"{dname}: test acc      = {accuracy_score(y_true, y_pred):.4f}")
    if irf_val is not None:
        print(f"{dname}: IRF           = {irf_val:.4f}")

    # 1) Loss curves
    try:
        if train_loss and val_loss:
            plt.figure()
            epochs = range(1, len(train_loss) + 1)
            plt.plot(epochs, train_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} – Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting loss for {dname}: {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        if train_acc and val_acc:
            plt.figure()
            epochs = range(1, len(train_acc) + 1)
            plt.plot(epochs, train_acc, label="Train", color="green")
            plt.plot(epochs, val_acc, label="Validation", color="orange")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname} – Training vs Validation Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_accuracy_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting accuracy for {dname}: {e}")
        plt.close()

    # 3) Confusion matrix
    try:
        if y_true.size and y_pred.size:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.title(
                f"{dname} – Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center")
            plt.colorbar(im)
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dname}: {e}")
        plt.close()

    # 4) IRF vs Val-Acc bar
    try:
        if irf_val is not None and len(val_acc):
            plt.figure()
            bars = ["IRF", "Val Acc"]
            values = [irf_val, val_acc[-1]]
            plt.bar(bars, values, color=["purple", "grey"])
            plt.ylim(0, 1)
            plt.title(f"{dname} – IRF vs Validation Accuracy")
            fname = os.path.join(working_dir, f"{dname}_irf_vs_valacc.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting IRF bar for {dname}: {e}")
        plt.close()
