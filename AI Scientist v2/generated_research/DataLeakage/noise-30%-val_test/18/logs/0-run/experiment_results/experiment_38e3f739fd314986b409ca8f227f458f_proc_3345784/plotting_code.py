import matplotlib.pyplot as plt
import numpy as np
import os

# -------- working dir --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    key = ("SinusoidalPositionalEmbedding", "SPR_BENCH")
    try:
        ed = experiment_data[key[0]][key[1]]
    except KeyError:
        print("Requested keys not found in experiment_data.")
        ed = None

    if ed:
        epochs = np.arange(1, len(ed["losses"]["train"]) + 1)

        # ---- 1. loss curves ----
        try:
            plt.figure()
            plt.plot(epochs, ed["losses"]["train"], label="Train")
            plt.plot(epochs, ed["losses"]["val"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR_BENCH: Training vs Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating loss plot: {e}")
            plt.close()

        # ---- 2. MCC curves ----
        try:
            plt.figure()
            plt.plot(epochs, ed["metrics"]["train_MCC"], label="Train MCC")
            plt.plot(epochs, ed["metrics"]["val_MCC"], label="Val MCC")
            plt.xlabel("Epoch")
            plt.ylabel("Matthews CorrCoef")
            plt.title("SPR_BENCH: Training vs Validation MCC")
            plt.legend()
            fname = os.path.join(working_dir, "SPR_BENCH_MCC_curves.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating MCC plot: {e}")
            plt.close()

        # ---- 3. confusion matrix (test set) ----
        try:
            preds = np.array(ed.get("predictions", []))
            gts = np.array(ed.get("ground_truth", []))
            if preds.size and gts.size and preds.size == gts.size:
                num_cls = int(max(preds.max(), gts.max()) + 1)
                cm = np.zeros((num_cls, num_cls), dtype=int)
                for p, g in zip(preds, gts):
                    cm[g, p] += 1

                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xlabel("Predicted")
                plt.ylabel("Ground Truth")
                plt.title(
                    "SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
                )
                for i in range(num_cls):
                    for j in range(num_cls):
                        plt.text(
                            j,
                            i,
                            cm[i, j],
                            ha="center",
                            va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                        )
                fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
                plt.savefig(fname)
                plt.close()
                print(f"Saved {fname}")
            else:
                print("Predictions or ground truth missing, skipping confusion matrix.")
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()
