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

all_final_val_acc = []
all_dnames = []

# ------------------ ITERATE AND PLOT -------------------------------
for dname, ddict in experiment_data.items():
    losses = ddict.get("losses", {})
    metrics = ddict.get("metrics", {})
    y_pred = np.array(ddict.get("predictions", []))
    y_true = np.array(ddict.get("ground_truth", []))

    epochs = list(range(1, len(losses.get("train", [])) + 1))
    final_val_acc = metrics.get("val", [np.nan])[-1] if metrics.get("val") else np.nan
    final_irf = metrics.get("IRF", [np.nan])[-1] if metrics.get("IRF") else np.nan
    print(f"{dname}: final val acc={final_val_acc:.3f}, final IRF={final_irf:.3f}")

    all_final_val_acc.append(final_val_acc)
    all_dnames.append(dname)

    # 1) Loss curves
    try:
        if losses.get("train") and losses.get("val"):
            plt.figure()
            plt.plot(epochs, losses["train"], marker="o", label="Train")
            plt.plot(epochs, losses["val"], marker="s", label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} – Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_loss_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting loss curve for {dname}: {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        if metrics.get("train") and metrics.get("val"):
            plt.figure()
            plt.plot(epochs, metrics["train"], marker="o", label="Train")
            plt.plot(epochs, metrics["val"], marker="s", label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"{dname} – Training vs Validation Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_accuracy_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting accuracy curve for {dname}: {e}")
        plt.close()

    # 3) IRF curve
    try:
        irf = metrics.get("IRF", [])
        if len(irf):
            plt.figure()
            plt.plot(epochs, irf, marker="d", color="purple")
            plt.xlabel("Epoch")
            plt.ylabel("IRF")
            plt.title(f"{dname} – Interpretation Fidelity (IRF)")
            fname = os.path.join(working_dir, f"{dname}_irf_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting IRF curve for {dname}: {e}")
        plt.close()

    # 4) Confusion matrix
    try:
        if y_true.size and y_pred.size:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_true, y_pred)
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.title(
                f"{dname} – Confusion Matrix\nLeft: Ground Truth, Right: Generated Samples"
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            for (i, j), v in np.ndenumerate(cm):
                plt.text(j, i, str(v), ha="center", va="center")
            plt.colorbar(im)
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix_viz.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dname}: {e}")
        plt.close()

# --------------- Cross-dataset comparison plot ---------------------
try:
    if len(all_dnames) >= 2:
        plt.figure()
        plt.bar(all_dnames, all_final_val_acc, color="teal")
        plt.ylabel("Final Validation Accuracy")
        plt.title("Dataset Comparison – Final Validation Accuracy")
        fname = os.path.join(working_dir, "comparison_final_val_accuracy.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error plotting comparison figure: {e}")
    plt.close()
