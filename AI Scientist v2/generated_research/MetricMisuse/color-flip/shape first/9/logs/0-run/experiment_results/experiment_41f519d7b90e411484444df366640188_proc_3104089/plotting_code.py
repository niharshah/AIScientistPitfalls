import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- I/O ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dname, ddata in experiment_data.items():
    losses = ddata.get("losses", {})
    metrics = ddata.get("metrics", {})
    preds = np.array(ddata.get("predictions", []))
    gtruth = np.array(ddata.get("ground_truth", []))

    # --------- accuracy + best metric ----------
    try:
        acc = (preds == gtruth).mean() if preds.size else float("nan")
        best_metric = max(metrics.get("val", [float("nan")]))
        print(f"{dname}: final accuracy={acc:.4f}, best_CompWA={best_metric:.4f}")
    except Exception as e:
        print(f"Error computing summary stats for {dname}: {e}")

    # --------- 1. loss curves ----------
    try:
        plt.figure()
        epochs = np.arange(1, len(losses.get("train", [])) + 1)
        plt.plot(epochs, losses.get("train", []), label="train")
        plt.plot(epochs, losses.get("val", []), label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname} – Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{dname}_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {dname}: {e}")
        plt.close()

    # --------- 2. validation CompWA curve ----------
    try:
        plt.figure()
        epochs = np.arange(1, len(metrics.get("val", [])) + 1)
        plt.plot(epochs, metrics.get("val", []), marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        plt.title(f"{dname} – Validation CompWA over Epochs")
        fname = os.path.join(working_dir, f"{dname}_CompWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot for {dname}: {e}")
        plt.close()

    # --------- 3. confusion matrix ----------
    try:
        if preds.size and gtruth.size:
            num_cls = int(max(preds.max(), gtruth.max())) + 1
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for t, p in zip(gtruth, preds):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dname} – Confusion Matrix\nrows=GT, cols=Pred")
            fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()
