import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# iterate through each dataset stored
for dname, dct in experiment_data.items():
    epochs = dct.get("epochs", None)
    train_losses = dct.get("losses", {}).get("train", [])
    val_losses = dct.get("losses", {}).get("val", [])
    val_compwa = dct.get("metrics", {}).get("val_compwa", [])
    preds = np.array(dct.get("predictions", []))
    gts = np.array(dct.get("ground_truth", []))

    # 1) Loss curves
    try:
        if len(epochs) and len(train_losses) and len(val_losses):
            plt.figure()
            plt.plot(epochs, train_losses, label="Train Loss")
            plt.plot(epochs, val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dname} – Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_loss_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # 2) CompWA curve
    try:
        if len(epochs) and len(val_compwa):
            plt.figure()
            plt.plot(epochs, val_compwa, label="Val CompWA")
            plt.xlabel("Epoch")
            plt.ylabel("CompWA")
            plt.title(f"{dname} – Validation Complexity-Weighted Accuracy")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_compwa_curve.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot for {dname}: {e}")
        plt.close()

    # 3) Confusion matrix heat-map (optional)
    try:
        if preds.size and gts.size:
            classes = sorted(list(set(gts) | set(preds)))
            conf = np.zeros((len(classes), len(classes)), dtype=int)
            for t, p in zip(gts, preds):
                conf[classes.index(t), classes.index(p)] += 1
            plt.figure()
            im = plt.imshow(conf, cmap="Blues")
            plt.colorbar(im)
            plt.xticks(range(len(classes)), classes)
            plt.yticks(range(len(classes)), classes)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dname} – Confusion Matrix (Test Set)")
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()
