import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import itertools

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH_reasoning" in experiment_data:
    rec = experiment_data["SPR_BENCH_reasoning"]
    epochs = rec.get("epochs", [])
    # -------- Figure 1 : Macro-F1 curves ---------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        fig.suptitle(
            "SPR_BENCH_reasoning Macro-F1 over Epochs\nLeft: Train  Right: Validation",
            fontsize=14,
        )
        axes[0].plot(epochs, rec["metrics"]["train_macro_f1"], label="train")
        axes[1].plot(
            epochs, rec["metrics"]["val_macro_f1"], label="val", color="orange"
        )
        for ax, ttl in zip(axes, ["Train Macro-F1", "Validation Macro-F1"]):
            ax.set_title(ttl)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Macro-F1")
            ax.set_ylim(0, 1)
            ax.legend()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_reasoning_macro_f1_curves.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating Macro-F1 plot: {e}")
        plt.close()

    # -------- Figure 2 : Loss curves -------------------------------------
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        fig.suptitle(
            "SPR_BENCH_reasoning Cross-Entropy Loss over Epochs\nLeft: Train  Right: Validation",
            fontsize=14,
        )
        axes[0].plot(epochs, rec["losses"]["train"], label="train")
        axes[1].plot(epochs, rec["losses"]["val"], label="val", color="orange")
        for ax, ttl in zip(axes, ["Train Loss", "Validation Loss"]):
            ax.set_title(ttl)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_reasoning_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot: {e}")
        plt.close()

    # -------- Figure 3 : Confusion matrix --------------------------------
    try:
        y_true = np.array(rec.get("ground_truth", []))
        y_pred = np.array(rec.get("predictions", []))
        if y_true.size and y_pred.size:
            labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            fig = plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("SPR_BENCH_reasoning Test Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45)
            plt.yticks(tick_marks, labels)
            thresh = cm.max() / 2.0
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "SPR_BENCH_reasoning_confusion_matrix.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating Confusion Matrix plot: {e}")
        plt.close()

    # -------- Console summary --------------------------------------------
    print(
        f"\nFinal Test Metrics for SPR_BENCH_reasoning:\n  Loss       : {rec.get('test_loss', None):.4f}\n  Macro-F1   : {rec.get('test_macro_f1', None):.4f}"
    )
