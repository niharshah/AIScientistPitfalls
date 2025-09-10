import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    ds_name = "SPR_BENCH"
    ds = experiment_data[ds_name]

    # --------------- 1. loss curve ----------------------------
    try:
        tr_epochs, tr_losses = zip(*ds["losses"]["train"])
        val_epochs, val_losses = zip(*ds["losses"]["val"])

        plt.figure()
        plt.plot(tr_epochs, tr_losses, label="Train")
        plt.plot(val_epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{ds_name} Loss Curve\nLeft: Train, Right: Validation")
        plt.legend()
        fname = f"loss_curve_{ds_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # --------------- 2. metric curves -------------------------
    try:
        # metrics stored as list of (epoch, dict)
        metr_list = ds["metrics"]["val"]
        epochs = [t for t, _ in metr_list]
        cwa = [d["CWA"] for _, d in metr_list]
        swa = [d["SWA"] for _, d in metr_list]
        pcwa = [d["PCWA"] for _, d in metr_list]

        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, pcwa, label="PCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{ds_name} Validation Metrics Over Epochs\nCWA, SWA, PCWA")
        plt.legend()
        fname = f"val_metric_curves_{ds_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # --------------- 3. confusion matrix ----------------------
    try:
        gts = ds["ground_truth"]
        prs = ds["predictions"]
        # create label ordering
        labels = sorted(set(gts) | set(prs))
        lab2idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(gts, prs):
            cm[lab2idx[t], lab2idx[p]] += 1

        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"{ds_name} Confusion Matrix\nLeft: GT, Right: Predicted")
        fname = f"confusion_matrix_{ds_name}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
