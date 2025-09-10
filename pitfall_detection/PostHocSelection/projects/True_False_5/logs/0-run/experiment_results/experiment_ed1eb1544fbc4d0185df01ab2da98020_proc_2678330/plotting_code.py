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
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


def shape_weighted_accuracy(seqs, y_true, y_pred):
    def count_shape_variety(sequence: str) -> int:
        return len(
            set(tok.split()[0] if tok else "" for tok in sequence.strip().split())
        )

    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if yt == yp else 0 for w, yt, yp in zip(weights, y_true, y_pred)]
    return sum(correct) / (sum(weights) if sum(weights) else 1.0)


for dname, rec in experiment_data.items():
    # ---- extract arrays safely ----
    train_loss = np.asarray(rec.get("losses", {}).get("train", []))
    val_loss = np.asarray(rec.get("losses", {}).get("val", []))
    train_swa = np.asarray(rec.get("metrics", {}).get("train_swa", []))
    val_swa = np.asarray(rec.get("metrics", {}).get("val_swa", []))
    preds = np.asarray(rec.get("predictions", []))
    gts = np.asarray(rec.get("ground_truth", []))
    epochs = np.arange(1, len(train_loss) + 1)

    # --------- plot 1: loss curves ---------
    try:
        if train_loss.size and val_loss.size:
            plt.figure()
            plt.plot(epochs, train_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname}: Train vs Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dname}: {e}")
        plt.close()

    # --------- plot 2: SWA curves ---------
    try:
        if train_swa.size and val_swa.size:
            plt.figure()
            plt.plot(epochs, train_swa, label="Train SWA")
            plt.plot(epochs, val_swa, label="Validation SWA")
            plt.xlabel("Epoch")
            plt.ylabel("Shape-Weighted Accuracy")
            plt.title(f"{dname}: Train vs Validation SWA")
            plt.legend()
            plt.savefig(os.path.join(working_dir, f"{dname}_swa_curves.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating SWA plot for {dname}: {e}")
        plt.close()

    # --------- plot 3: Test accuracy bar ---------
    try:
        if preds.size and gts.size:
            test_acc = (preds == gts).mean()
            plt.figure()
            plt.bar(["Accuracy"], [test_acc])
            plt.ylim(0, 1)
            plt.title(f"{dname}: Test Accuracy")
            plt.savefig(os.path.join(working_dir, f"{dname}_test_accuracy.png"))
            plt.close()

            # also compute and print SWA on test if sequences available
            seqs = rec.get("test_sequences", [])
            if len(seqs) == len(preds):
                test_swa = shape_weighted_accuracy(seqs, gts, preds)
            else:
                test_swa = np.nan
            print(f"{dname} - Test Accuracy: {test_acc:.4f} | Test SWA: {test_swa:.4f}")
        else:
            print(f"{dname} - No prediction data available.")
    except Exception as e:
        print(f"Error creating accuracy plot for {dname}: {e}")
        plt.close()
