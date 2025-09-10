import matplotlib.pyplot as plt
import numpy as np
import os

# working directory for outputs
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dname, rec in experiment_data.items():
    epochs = rec.get("epochs", [])
    # -------- Figure 1 : Macro-F1 curves ---------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        fig.suptitle(
            f"{dname} Macro-F1 over Epochs\nLeft: Train  Right: Validation", fontsize=14
        )
        axes[0].plot(epochs, rec["metrics"]["train_macro_f1"], label="train")
        axes[1].plot(epochs, rec["metrics"]["val_macro_f1"], label="val")
        for ax, ttl in zip(axes, ["Train Macro-F1", "Validation Macro-F1"]):
            ax.set_title(ttl)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Macro-F1")
            ax.legend()
        fname = os.path.join(working_dir, f"{dname}_macro_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Macro-F1 plot ({dname}): {e}")
        plt.close()

    # -------- Figure 2 : Loss curves -------------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        fig.suptitle(
            f"{dname} Cross-Entropy Loss over Epochs\nLeft: Train  Right: Validation",
            fontsize=14,
        )
        axes[0].plot(epochs, rec["losses"]["train"], label="train")
        axes[1].plot(epochs, rec["losses"]["val"], label="val")
        for ax, ttl in zip(axes, ["Train Loss", "Validation Loss"]):
            ax.set_title(ttl)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot ({dname}): {e}")
        plt.close()

    # -------- Figure 3 : Confusion matrix --------------------------------
    try:
        preds = np.asarray(rec.get("predictions", []))
        trues = np.asarray(rec.get("ground_truth", []))
        if preds.size and trues.size and preds.size == trues.size:
            num_classes = int(max(trues.max(), preds.max()) + 1)
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for p, t in zip(preds, trues):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            plt.title(f"{dname} Test Confusion Matrix")
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            ticks = np.arange(num_classes)
            plt.xticks(ticks)
            plt.yticks(ticks)
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Error creating Confusion Matrix ({dname}): {e}")
        plt.close()

    # -------- Console summary --------------------------------------------
    test_f1 = rec.get("test_macro_f1")
    test_loss = rec.get("test_loss")
    if test_f1 is not None:
        print(f"{dname}  Test Macro-F1: {test_f1:.4f}  Test Loss: {test_loss:.4f}")
