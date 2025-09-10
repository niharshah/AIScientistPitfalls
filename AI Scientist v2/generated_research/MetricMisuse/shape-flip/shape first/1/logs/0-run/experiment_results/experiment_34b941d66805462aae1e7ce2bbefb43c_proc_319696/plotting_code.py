import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr = experiment_data["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr = None

if spr is not None:
    epochs = np.arange(1, len(spr["metrics"]["train_acc"]) + 1)

    # --------- train / val accuracy --------------------------------
    try:
        plt.figure()
        plt.plot(epochs, spr["metrics"]["train_acc"], label="Train Acc")
        plt.plot(epochs, spr["metrics"]["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH – Training vs Validation Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # --------- train / val loss ------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, spr["losses"]["train"], label="Train Loss")
        plt.plot(epochs, spr["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH – Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_train_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # --------- URA curve -------------------------------------------
    try:
        ura = spr["metrics"]["URA"]
        plt.figure()
        plt.plot(epochs, ura, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Unseen Rule Accuracy (URA)")
        plt.title("SPR_BENCH – URA Across Epochs")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_URA_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating URA plot: {e}")
        plt.close()

    # --------- confusion matrix ------------------------------------
    try:
        preds = spr["predictions"]
        gold = spr["ground_truth"]
        labels = sorted(list(set(gold) | set(preds)))
        L = len(labels)
        label2idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((L, L), dtype=int)
        for g, p in zip(gold, preds):
            cm[label2idx[g], label2idx[p]] += 1

        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        plt.colorbar(im, ax=ax)
        plt.title(
            "SPR_BENCH – Confusion Matrix\n(Left: Ground Truth, Right: Predicted)"
        )
        plt.ylabel("Ground Truth")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
