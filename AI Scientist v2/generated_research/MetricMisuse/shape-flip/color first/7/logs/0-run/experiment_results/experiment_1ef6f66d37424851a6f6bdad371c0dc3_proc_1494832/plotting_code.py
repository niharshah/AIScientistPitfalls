import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ---------- helper for CompWA ----------
def count_color_variety(seq):
    return len(set(tok[1:] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def compWA(seqs, y_true, y_pred):
    weights = np.array([count_color_variety(s) + count_shape_variety(s) for s in seqs])
    correct = (np.array(y_true) == np.array(y_pred)).astype(int)
    return (weights * correct).sum() / max(1, weights.sum())


# ---------- plotting ----------
for ds_name, ds in experiment_data.items():
    # guard against missing keys
    losses = ds.get("losses", {})
    metrics = ds.get("metrics", {})
    epochs = np.arange(1, len(losses.get("train", [])) + 1)

    # 1) Loss curves -------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_name} Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        plt.close()

    # 2) Accuracy curves --------------------------------------------
    try:
        tr_acc = [m["acc"] for m in metrics.get("train", [])]
        val_acc = [m["acc"] for m in metrics.get("val", [])]
        plt.figure()
        plt.plot(epochs, tr_acc, label="Train Acc")
        plt.plot(epochs, val_acc, label="Validation Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_name} Accuracy Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds_name}: {e}")
        plt.close()

    # 3) CompWA curve (validation) ----------------------------------
    try:
        val_cowa = [m["CompWA"] for m in metrics.get("val", [])]
        plt.figure()
        plt.plot(epochs, val_cowa, label="Validation CompWA")
        plt.xlabel("Epoch")
        plt.ylabel("CompWA")
        plt.title(f"{ds_name} CompWA Curve")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_compwa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CompWA plot for {ds_name}: {e}")
        plt.close()

    # ---------- evaluation metrics on stored predictions ------------
    try:
        preds = np.array(ds["predictions"])
        gts = np.array(ds["ground_truth"])
        seqs = np.array(ds["sequences"])
        if len(preds):
            test_acc = (preds == gts).mean()
            test_cowa = compWA(seqs, gts, preds)
            print(
                f"{ds_name} -- Test Accuracy: {test_acc:.3f} | Test CompWA: {test_cowa:.3f}"
            )
    except Exception as e:
        print(f"Error computing evaluation metrics for {ds_name}: {e}")
