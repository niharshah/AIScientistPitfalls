import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ SETUP ---------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ LOAD ----------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------ PLOTS ---------------------------
for dname, ddict in experiment_data.items():
    losses = ddict.get("losses", {})
    metrics = ddict.get("metrics", {})
    y_pred = np.array(ddict.get("predictions", []))
    y_true = np.array(ddict.get("ground_truth", []))
    epochs = np.arange(1, len(metrics.get("train_acc", [])) + 1)

    # 1) Loss curves -------------------------------------------------
    try:
        tr_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        if tr_loss and val_loss:
            plt.figure()
            plt.plot(epochs, tr_loss, marker="o", label="Train")
            plt.plot(epochs, val_loss, marker="s", label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} – Train vs Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting loss curves for {dname}: {e}")
        plt.close()

    # 2) Accuracy curves --------------------------------------------
    try:
        tr_acc = metrics.get("train_acc", [])
        val_acc = metrics.get("val_acc", [])
        if tr_acc and val_acc:
            plt.figure()
            plt.plot(epochs, tr_acc, marker="o", label="Train")
            plt.plot(epochs, val_acc, marker="s", label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname} – Train vs Validation Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_accuracy_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting accuracy curves for {dname}: {e}")
        plt.close()

    # 3) IRF over epochs --------------------------------------------
    try:
        irf_vals = metrics.get("IRF", [])
        if irf_vals:
            plt.figure()
            plt.plot(epochs, irf_vals, marker="d", color="purple")
            plt.xlabel("Epoch")
            plt.ylabel("IRF")
            plt.title(f"{dname} – Interpretable Replication Fidelity (IRF)")
            plt.savefig(os.path.join(working_dir, f"{dname}_irf_curve.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting IRF for {dname}: {e}")
        plt.close()

    # 4) IRF vs Val-Acc scatter -------------------------------------
    try:
        if val_acc and irf_vals:
            plt.figure()
            plt.scatter(val_acc, irf_vals, c="teal")
            for i, (x, y) in enumerate(zip(val_acc, irf_vals)):
                plt.text(x, y, str(i + 1))
            plt.xlabel("Validation Accuracy")
            plt.ylabel("IRF")
            plt.title(f"{dname} – IRF vs Validation Accuracy")
            plt.savefig(os.path.join(working_dir, f"{dname}_irf_vs_valacc.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting IRF-vs-Acc for {dname}: {e}")
        plt.close()

    # 5) Confusion matrix -------------------------------------------
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
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dname}: {e}")
        plt.close()

    # ------------------ METRIC PRINTS ------------------------------
    try:
        print(
            f"{dname}: test_acc={metrics.get('test_acc', 'NA'):.3f}, "
            f"IRF_test={metrics.get('IRF_test', 'NA'):.3f}"
        )
    except Exception:
        pass
