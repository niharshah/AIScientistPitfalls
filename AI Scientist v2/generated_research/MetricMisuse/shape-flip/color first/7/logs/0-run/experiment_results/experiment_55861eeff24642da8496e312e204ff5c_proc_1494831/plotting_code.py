import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper for CompWA ----------
def complexity_weight(seq: str) -> int:
    toks = seq.strip().split()
    color_var = len(set(t[1:] for t in toks if len(t) > 1))
    shape_var = len(set(t[0] for t in toks if t))
    return color_var + shape_var


# ---------- iterate over datasets ----------
for ds_name, ds in experiment_data.items():
    # pull history lists
    epochs = np.arange(1, len(ds["losses"]["train"]) + 1)
    train_loss = ds["losses"]["train"]
    val_loss = ds["losses"]["val"]
    train_acc = [m.get("acc", np.nan) for m in ds["metrics"]["train"]]
    val_acc = [m.get("acc", np.nan) for m in ds["metrics"]["val"]]
    val_cwa = [m.get("CompWA", np.nan) for m in ds["metrics"]["val"]]

    # 1) Loss curve
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # 2) Accuracy curve
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} Accuracy Curves\nLeft: Train, Right: Validation")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()

    # 3) CompWA curve
    try:
        plt.figure()
        plt.plot(epochs, val_cwa, label="Validation CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        plt.title(f"{ds_name} Complexity-Weighted Accuracy\nValidation Set")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_compwa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot for {ds_name}: {e}")
        plt.close()

    # ---------- evaluation metrics & confusion matrix ----------
    preds = np.array(ds.get("predictions", []))
    gts = np.array(ds.get("ground_truth", []))
    seqs = np.array(ds.get("sequences", []))
    if len(preds) and len(gts) and len(seqs):
        try:
            test_acc = (preds == gts).mean()
            weights = np.array([complexity_weight(s) for s in seqs])
            compwa = (weights * (preds == gts)).sum() / weights.sum()
            print(
                f"{ds_name}  Test Accuracy: {test_acc:.3f} | Test CompWA: {compwa:.3f}"
            )
        except Exception as e:
            print(f"Error computing metrics for {ds_name}: {e}")

        # 4) Confusion matrix heat-map
        try:
            classes = np.unique(np.concatenate([preds, gts]))
            n_cls = len(classes)
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for p, g in zip(preds, gts):
                cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{ds_name} Confusion Matrix")
            for i in range(n_cls):
                for j in range(n_cls):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.savefig(os.path.join(working_dir, f"{ds_name}_confusion_matrix.png"))
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {ds_name}: {e}")
            plt.close()
