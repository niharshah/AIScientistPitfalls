import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# assume single dataset key
if experiment_data:
    ds_name = list(experiment_data.keys())[0]
    ds = experiment_data[ds_name]
    train_losses = ds["losses"].get("train", [])
    val_losses = ds["losses"].get("val", [])
    val_metric = ds["metrics"].get("val", [])
    preds = np.array(ds.get("predictions", []))
    gts = np.array(ds.get("ground_truth", []))

    # ---- Plot 1: Loss curves
    try:
        plt.figure()
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Training vs Validation Loss")
        plt.legend()
        fname = f"{ds_name}_loss_curves.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot: {e}")
        plt.close()

    # ---- Plot 2: Validation CoWA
    try:
        plt.figure()
        plt.plot(epochs, val_metric, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title(f"{ds_name} Validation CoWA over Epochs")
        fname = f"{ds_name}_val_cowa.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating CoWA plot: {e}")
        plt.close()

    # ---- Plot 3: Confusion matrix
    try:
        if preds.size and gts.size:
            num_labels = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_labels, num_labels), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"{ds_name} Confusion Matrix")
            fname = f"{ds_name}_confusion_matrix.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating Confusion Matrix plot: {e}")
        plt.close()

    # ---- print evaluation metric
    try:
        # recompute CoWA
        def count_color_variety(sequence: str) -> int:
            return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))

        def count_shape_variety(sequence: str) -> int:
            return len(set(tok[0] for tok in sequence.strip().split() if tok))

        seqs = ds.get("ground_truth", [])  # actually we don't have sequences; skip
        if "sequences" in ds:
            seqs = ds["sequences"]
        if isinstance(seqs, list) and len(seqs) == len(preds):
            weights = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
            correct = [w if t == p else 0 for w, t, p in zip(weights, gts, preds)]
            cowa = sum(correct) / sum(weights) if sum(weights) > 0 else 0.0
        else:
            cowa = np.mean(preds == gts) if gts.size else 0.0
        print(f"Test set COWA (recomputed): {cowa:.4f}")
    except Exception as e:
        print(f"Error computing/printing metric: {e}")
