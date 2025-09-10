import matplotlib.pyplot as plt
import numpy as np
import os

# --- paths ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load experiment data ---
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run_data = experiment_data["no_global_stats"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run_data = None

if run_data:
    epochs = np.arange(1, len(run_data["losses"]["train"]) + 1)
    train_loss = np.array(run_data["losses"]["train"])
    val_loss = np.array(run_data["losses"]["val"])
    train_swa = np.array(run_data["metrics"]["train_swa"])
    val_swa = np.array(run_data["metrics"]["val_swa"])
    preds = np.array(run_data["predictions"])
    gtruth = np.array(run_data["ground_truth"])

    # ---------- 1) Loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("spr_bench Loss Curves\nTrain vs Validation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(working_dir, "spr_bench_loss_curve.png"))
    except Exception as e:
        print(f"Error creating loss curve: {e}")
    finally:
        plt.close()

    # ---------- 2) SWA curves ----------
    try:
        plt.figure()
        plt.plot(epochs, train_swa, label="Train SWA")
        plt.plot(epochs, val_swa, label="Validation SWA")
        plt.xlabel("Epoch")
        plt.ylabel("Shape-Weighted Accuracy")
        plt.title("spr_bench SWA Curves\nTrain vs Validation")
        plt.legend()
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(working_dir, "spr_bench_swa_curve.png"))
    except Exception as e:
        print(f"Error creating SWA curve: {e}")
    finally:
        plt.close()

    # ---------- 3) Confusion matrix ----------
    try:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gtruth, preds):
            cm[int(t), int(p)] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title("spr_bench Confusion Matrix\nRows: Ground Truth, Cols: Predictions")
        plt.colorbar()
        plt.savefig(os.path.join(working_dir, "spr_bench_confusion_matrix.png"))
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
    finally:
        plt.close()

    print(f"Plots saved to {working_dir}")
