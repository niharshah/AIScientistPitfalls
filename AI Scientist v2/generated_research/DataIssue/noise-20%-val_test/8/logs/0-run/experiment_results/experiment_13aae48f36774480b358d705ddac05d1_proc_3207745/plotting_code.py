import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ PREP ------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ LOAD DATA -------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------ PLOT PER DATASET ------------------
test_accs = {}
for dname, ddict in experiment_data.items():
    losses = ddict.get("losses", {})
    metrics = ddict.get("metrics", {})
    y_true = np.array(ddict.get("ground_truth", []))
    # accommodate both "test_predictions" and older "predictions"
    y_pred = np.array(ddict.get("test_predictions", ddict.get("predictions", [])))

    # Extract series
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    train_acc = metrics.get("train_acc", [])
    val_acc = metrics.get("val_acc", [])
    test_acc = metrics.get("test_acc", None)
    irf = metrics.get("IRF", None)
    if test_acc is not None:
        test_accs[dname] = test_acc

    # Text summary
    print(f"\n=== {dname} ===")
    if train_acc:
        print(f"Final train acc: {train_acc[-1]:.3f}")
    if val_acc:
        print(f"Final val   acc: {val_acc[-1]:.3f}")
    if test_acc is not None:
        print(f"Test  acc: {test_acc:.3f}")
    if irf is not None:
        print(f"IRF      : {irf:.3f}")

    # 1) Loss curves -------------------------------------------------
    try:
        if train_loss or val_loss:
            plt.figure()
            if train_loss:
                plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train")
            if val_loss:
                plt.plot(range(1, len(val_loss) + 1), val_loss, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} – Loss Curves")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting loss for {dname}: {e}")
        plt.close()

    # 2) Accuracy curves --------------------------------------------
    try:
        if train_acc or val_acc:
            plt.figure()
            if train_acc:
                plt.plot(range(1, len(train_acc) + 1), train_acc, label="Train")
            if val_acc:
                plt.plot(range(1, len(val_acc) + 1), val_acc, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname} – Accuracy Curves")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_accuracy_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting accuracy for {dname}: {e}")
        plt.close()

    # 3) Confusion matrix -------------------------------------------
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
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dname}: {e}")
        plt.close()

# ------------------ COMPARISON PLOT -------------------
try:
    if len(test_accs) > 1:
        plt.figure()
        names, accs = zip(*test_accs.items())
        plt.bar(range(len(accs)), accs, tick_label=names, color="orange")
        plt.ylabel("Test Accuracy")
        plt.title("Dataset Comparison – Test Accuracy")
        fname = os.path.join(working_dir, "comparison_test_accuracy.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating comparison plot: {e}")
    plt.close()
