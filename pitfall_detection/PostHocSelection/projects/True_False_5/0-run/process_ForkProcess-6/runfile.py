import matplotlib.pyplot as plt
import numpy as np
import os

# ----- paths -----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load experiment data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp_rec = experiment_data["epochs_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_rec = None

if exp_rec is not None:
    # extract arrays
    train_loss = np.asarray(exp_rec["losses"]["train"])
    val_loss = np.asarray(exp_rec["losses"]["val"])
    train_rcwa = np.asarray(exp_rec["metrics"]["train_rcwa"])
    val_rcwa = np.asarray(exp_rec["metrics"]["val_rcwa"])
    preds = np.asarray(exp_rec["predictions"])
    gts = np.asarray(exp_rec["ground_truth"])
    epochs = np.arange(1, len(train_loss) + 1)

    # --------- plot 1: loss curves ---------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Train vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # --------- plot 2: RCWA curves ---------
    try:
        plt.figure()
        plt.plot(epochs, train_rcwa, label="Train RCWA")
        plt.plot(epochs, val_rcwa, label="Validation RCWA")
        plt.xlabel("Epoch")
        plt.ylabel("RCWA")
        plt.title("SPR_BENCH: Train vs Validation RCWA")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_rcwa_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating RCWA plot: {e}")
        plt.close()

    # --------- plot 3: Test accuracy bar ---------
    try:
        test_acc = (preds == gts).mean() if len(preds) else np.nan
        plt.figure()
        plt.bar(["Accuracy"], [test_acc])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Test Accuracy")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ----- print evaluation metric -----
    print(f"Test Accuracy: {test_acc:.4f}")
