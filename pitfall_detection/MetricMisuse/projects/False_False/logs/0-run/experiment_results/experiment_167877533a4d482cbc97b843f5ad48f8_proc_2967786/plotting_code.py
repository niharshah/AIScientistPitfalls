import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------------------
# set up working dir and load experiment data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "contrastive_ft" in experiment_data:
    data = experiment_data["contrastive_ft"]
    train_loss = np.array(data["losses"]["train"])
    val_loss = np.array(data["losses"]["val"])
    train_scwa = np.array(data["metrics"]["SCWA_train"])
    val_scwa = np.array(data["metrics"]["SCWA_val"])
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])
    epochs = np.arange(1, len(train_loss) + 1)

    # ------------------ Plot 1: Loss curves ----------------------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss over Epochs\nLeft: Train, Right: Val")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_contrastive_ft_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------------------ Plot 2: SCWA curves ----------------------
    try:
        plt.figure()
        plt.plot(epochs, train_scwa, label="Train SCWA")
        plt.plot(epochs, val_scwa, label="Val SCWA")
        plt.xlabel("Epoch")
        plt.ylabel("SCWA")
        plt.title("SPR_BENCH SCWA over Epochs\nLeft: Train, Right: Val")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_contrastive_ft_SCWA_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating SCWA curve: {e}")
        plt.close()

    # ------------- Plot 3: Confusion-matrix heat-map -------------
    # limit to first 10 unique labels for clarity
    try:
        uniq_lbls = np.unique(np.concatenate([gts, preds]))[:10]
        lbl_map = {l: i for i, l in enumerate(uniq_lbls)}
        cm = np.zeros((len(uniq_lbls), len(uniq_lbls)), dtype=int)
        for gt, pr in zip(gts, preds):
            if gt in lbl_map and pr in lbl_map:
                cm[lbl_map[gt], lbl_map[pr]] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH Confusion Matrix (first 10 labels)")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_contrastive_ft_confusion.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------------- Print final metrics ------------------------
    print(f"Final Val Loss:  {val_loss[-1]:.4f}")
    print(f"Final Val SCWA:  {val_scwa[-1]:.4f}")
else:
    print("No contrastive_ft data found in experiment_data.")
