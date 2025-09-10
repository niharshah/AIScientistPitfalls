import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
exp_file = os.path.join(working_dir, "experiment_data.npy")

# ---------- load ----------
try:
    experiment_data = np.load(exp_file, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- iterate over experiments ----------
for exp_name, exp_dict in experiment_data.items():
    losses = exp_dict.get("losses", {})
    metrics = exp_dict.get("metrics", {})
    preds = np.array(exp_dict.get("predictions", []))
    gtruth = np.array(exp_dict.get("ground_truth", []))

    # 1. loss curves ---------------------------------------------------------
    try:
        tr_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        if tr_loss and val_loss:
            plt.figure()
            epochs = np.arange(1, len(tr_loss) + 1)
            plt.plot(epochs, tr_loss, label="train")
            plt.plot(epochs, val_loss, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{exp_name} – Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{exp_name}_loss_curves.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {exp_name}: {e}")
        plt.close()

    # 2. CWA curve -----------------------------------------------------------
    try:
        val_cwa = metrics.get("val", [])
        if val_cwa:
            plt.figure()
            epochs = np.arange(1, len(val_cwa) + 1)
            plt.plot(epochs, val_cwa, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Complexity-Weighted Accuracy")
            plt.title(f"{exp_name} – Validation CWA per Epoch")
            fname = os.path.join(working_dir, f"{exp_name}_CWA_curve.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating CWA curve for {exp_name}: {e}")
        plt.close()

    # 3. confusion matrix ----------------------------------------------------
    try:
        if preds.size and gtruth.size:
            uniq = sorted(set(np.concatenate([preds, gtruth]).tolist()))
            m = len(uniq)
            label2idx = {l: i for i, l in enumerate(uniq)}
            cm = np.zeros((m, m), dtype=int)
            for gt, pr in zip(gtruth, preds):
                cm[label2idx[gt], label2idx[pr]] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{exp_name} – Confusion Matrix\n(rows=GT, cols=Pred)")
            fname = os.path.join(working_dir, f"{exp_name}_confusion_matrix.png")
            plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {exp_name}: {e}")
        plt.close()

    # 4. print quick stats ---------------------------------------------------
    try:
        acc = (preds == gtruth).mean() if preds.size else float("nan")
        best_cwa = max(metrics.get("val", [float("nan")]))
        print(f"{exp_name}: Final acc={acc:.4f}, Best CWA={best_cwa:.4f}")
    except Exception as e:
        print(f"Error computing stats for {exp_name}: {e}")
