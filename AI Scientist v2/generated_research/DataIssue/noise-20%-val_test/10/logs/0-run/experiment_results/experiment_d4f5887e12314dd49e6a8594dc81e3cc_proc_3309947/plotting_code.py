import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def macro_f1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for lbl in labels:
        tp = np.sum((y_true == lbl) & (y_pred == lbl))
        fp = np.sum((y_true != lbl) & (y_pred == lbl))
        fn = np.sum((y_true == lbl) & (y_pred != lbl))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return np.mean(f1s)


try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dname, d in experiment_data.items():
    losses_tr = d["losses"]["train"]
    losses_val = d["losses"]["val"]
    f1_tr = d["metrics"]["train_f1"]
    f1_val = d["metrics"]["val_f1"]
    ia = d["metrics"]["Interpretable_Acc"]

    # 1) Loss curve
    try:
        plt.figure()
        epochs = np.arange(1, len(losses_tr) + 1)
        plt.plot(epochs, losses_tr, label="Train")
        plt.plot(epochs, losses_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname} Loss Curve\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dname}: {e}")
        plt.close()

    # 2) F1 curve
    try:
        plt.figure()
        plt.plot(epochs, f1_tr, label="Train")
        plt.plot(epochs, f1_val, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title(f"{dname} Macro-F1 Curve\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve for {dname}: {e}")
        plt.close()

    # 3) Interpretable Accuracy
    try:
        plt.figure()
        plt.plot(epochs, ia, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Interpretable Accuracy")
        plt.title(f"{dname} Interpretable Accuracy Over Epochs")
        fname = os.path.join(working_dir, f"{dname}_interpretable_acc.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Interpretable Accuracy plot for {dname}: {e}")
        plt.close()

    # Final test metric
    try:
        y_pred = np.array(d["predictions"])
        y_true = np.array(d["ground_truth"])
        final_f1 = macro_f1(y_true, y_pred)
        print(f"{dname} final test Macro-F1: {final_f1:.4f}")
    except Exception as e:
        print(f"Error computing final Macro-F1 for {dname}: {e}")
