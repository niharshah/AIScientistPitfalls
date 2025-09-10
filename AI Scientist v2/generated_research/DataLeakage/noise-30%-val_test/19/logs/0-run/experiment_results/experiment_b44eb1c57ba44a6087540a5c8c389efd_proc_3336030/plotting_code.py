import matplotlib.pyplot as plt
import numpy as np
import os

# ------------ paths & data loading -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

test_scores_for_bar = {}
if experiment_data is not None:
    for dname, logs in experiment_data.items():
        epochs = logs.get("epochs", [])
        tr_loss = logs.get("losses", {}).get("train", [])
        val_loss = logs.get("losses", {}).get("val", [])
        tr_mcc = logs.get("metrics", {}).get("train_MCC", [])
        val_mcc = logs.get("metrics", {}).get("val_MCC", [])
        preds = logs.get("predictions")
        gts = logs.get("ground_truth")
        test_mcc = logs.get("test_MCC")
        if test_mcc is not None:
            test_scores_for_bar[dname] = test_mcc
            print(f"{dname}  test_MCC = {test_mcc:.4f}")

        # -------- loss curves --------
        try:
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} — Train vs Val Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname.lower()}_loss_curve.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dname}: {e}")
            plt.close()

        # -------- MCC curves --------
        try:
            plt.figure()
            plt.plot(epochs, tr_mcc, label="Train")
            plt.plot(epochs, val_mcc, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("MCC")
            plt.title(f"{dname} — Train vs Val MCC")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname.lower()}_mcc_curve.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating MCC plot for {dname}: {e}")
            plt.close()

        # -------- confusion matrix (test) --------
        try:
            if preds is not None and gts is not None and len(preds) == len(gts):
                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(gts, preds)
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                plt.colorbar()
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(
                            j, i, cm[i, j], ha="center", va="center", color="black"
                        )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"{dname} — Confusion Matrix (Test)")
                fname = os.path.join(
                    working_dir, f"{dname.lower()}_confusion_matrix.png"
                )
                plt.tight_layout()
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {dname}: {e}")
            plt.close()

# -------- aggregate bar chart of test MCC --------
try:
    if test_scores_for_bar:
        plt.figure()
        names = list(test_scores_for_bar.keys())
        values = [test_scores_for_bar[n] for n in names]
        plt.bar(names, values, color="skyblue")
        plt.ylabel("Test MCC")
        plt.title("Test MCC by Dataset")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "aggregate_test_mcc_bar.png"))
        plt.close()
        print("Aggregated test MCC:", test_scores_for_bar)
except Exception as e:
    print(f"Error creating aggregate bar chart: {e}")
    plt.close()
