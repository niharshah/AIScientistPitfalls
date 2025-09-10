import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper ----------
def save_close(fig, fname):
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- iterate over datasets ----------
for dname, dct in experiment_data.items():
    metrics = dct.get("metrics", {})
    losses = dct.get("losses", {})
    preds = np.asarray(dct.get("predictions", []))
    gts = np.asarray(dct.get("ground_truth", []))
    rule_preds = np.asarray(dct.get("rule_predictions", []))

    epochs = np.arange(1, len(metrics.get("train_acc", [])) + 1)

    # 1) accuracy curves
    try:
        fig = plt.figure(figsize=(6, 4))
        plt.plot(epochs, metrics.get("train_acc", []), label="train")
        plt.plot(epochs, metrics.get("val_acc", []), linestyle="--", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dname}: Training vs Validation Accuracy")
        plt.legend()
        save_close(fig, os.path.join(working_dir, f"{dname}_acc_curves.png"))
    except Exception as e:
        print(f"Error creating accuracy curves for {dname}: {e}")
        plt.close()

    # 2) loss curves
    try:
        fig = plt.figure(figsize=(6, 4))
        plt.plot(epochs, losses.get("train", []), label="train")
        plt.plot(epochs, metrics.get("val_loss", []), linestyle="--", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dname}: Training vs Validation Loss")
        plt.legend()
        save_close(fig, os.path.join(working_dir, f"{dname}_loss_curves.png"))
    except Exception as e:
        print(f"Error creating loss curves for {dname}: {e}")
        plt.close()

    # 3) confusion matrix
    try:
        if preds.size and gts.size:
            classes = sorted(np.unique(np.concatenate([gts, preds])))
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            for gt, pr in zip(gts, preds):
                cm[gt, pr] += 1
            fig = plt.figure(figsize=(4, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046)
            plt.xticks(range(len(classes)), classes)
            plt.yticks(range(len(classes)), classes)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dname}: Confusion Matrix (Test)")
            for i in range(len(classes)):
                for j in range(len(classes)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
            save_close(fig, os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

    # 4) accuracy vs fidelity
    try:
        test_acc = metrics.get("test_acc", None)
        test_rfs = metrics.get("test_RFS", None)
        if test_acc is not None and test_rfs is not None:
            fig = plt.figure(figsize=(4, 3))
            plt.bar(
                ["Test Acc", "Rule Fidelity"],
                [test_acc, test_rfs],
                color=["green", "orange"],
            )
            plt.ylim(0, 1)
            plt.title(f"{dname}: Test Accuracy vs Rule Fidelity")
            save_close(fig, os.path.join(working_dir, f"{dname}_acc_vs_fidelity.png"))
            print(f"{dname}  TestAcc={test_acc:.3f}  RuleFidelity={test_rfs:.3f}")
    except Exception as e:
        print(f"Error creating acc vs fidelity bar for {dname}: {e}")
        plt.close()
