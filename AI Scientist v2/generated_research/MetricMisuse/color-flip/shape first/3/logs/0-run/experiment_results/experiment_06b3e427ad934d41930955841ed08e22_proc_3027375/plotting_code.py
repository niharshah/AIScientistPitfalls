import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    ds_name = "SPR_BENCH"
    rec = experiment_data[ds_name]
    losses = rec["losses"]
    metrics = rec["metrics"]
    aca = rec.get("aca", {})
    preds = np.array(rec.get("predictions", []))
    gts = np.array(rec.get("ground_truth", []))
    epochs = np.arange(1, max(len(losses["train"]), len(losses["val"])) + 1)
    # optional down-sampling for clarity
    if len(epochs) > 50:
        step = int(np.ceil(len(epochs) / 50))
        sel = slice(None, None, step)
    else:
        sel = slice(None)

    # ------------------------------------------------- Plot 1: loss curves
    try:
        plt.figure()
        plt.plot(epochs[sel], np.array(losses["train"])[sel], "--", label="Train Loss")
        plt.plot(epochs[sel], np.array(losses["val"])[sel], "-", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.title(f"{ds_name}: Training vs Validation Loss")
        plt.legend()
        fn = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 2: validation accuracy
    try:
        plt.figure()
        plt.plot(epochs[sel], np.array(metrics["val"])[sel], color="g")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title(f"{ds_name}: Validation Accuracy per Epoch")
        fn = os.path.join(working_dir, f"{ds_name}_val_accuracy.png")
        plt.savefig(fn)
        print(f"Saved {fn}")
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 3: validation ACA
    try:
        if "val" in aca and len(aca["val"]):
            plt.figure()
            plt.plot(epochs[sel], np.array(aca["val"])[sel], color="m")
            plt.xlabel("Epoch")
            plt.ylabel("Validation ACA")
            plt.title(f"{ds_name}: Validation ACA per Epoch")
            fn = os.path.join(working_dir, f"{ds_name}_val_ACA.png")
            plt.savefig(fn)
            print(f"Saved {fn}")
            plt.close()
    except Exception as e:
        print(f"Error creating ACA plot: {e}")
        plt.close()

    # ------------------------------------------------- Plot 4: confusion matrix
    try:
        if preds.size and gts.size:
            num_classes = len(np.unique(np.concatenate([preds, gts])))
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(num_classes), [f"Pred {i}" for i in range(num_classes)])
            plt.yticks(range(num_classes), [f"True {i}" for i in range(num_classes)])
            plt.title(f"{ds_name}: Confusion Matrix (Test Set)")
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fn = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
            plt.savefig(fn)
            print(f"Saved {fn}")
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
else:
    print("No experiment data to plot.")
