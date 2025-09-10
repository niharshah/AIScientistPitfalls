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
    experiment_data = None

if experiment_data:
    # accommodate the saved structure
    dataset_name = list(experiment_data["num_epochs"].keys())[0]
    ds = experiment_data["num_epochs"][dataset_name]

    epochs = np.arange(1, len(ds["losses"]["train"]) + 1)
    train_loss = ds["losses"]["train"]
    val_loss = ds["losses"]["val"]
    train_acc = [m["acc"] for m in ds["metrics"]["train"]]
    val_acc = [m["acc"] for m in ds["metrics"]["val"]]
    val_cowa = [m["cowa"] for m in ds["metrics"]["val"]]

    # ---------- plotting ----------
    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset_name} Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 2) Accuracy curves
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset_name} Accuracy Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 3) CoWA curve (validation only)
    try:
        plt.figure()
        plt.plot(epochs, val_cowa, label="Validation CoWA")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title(f"{dataset_name} CoWA Curve")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_cowa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CoWA plot: {e}")
        plt.close()

    # ---------- evaluation metrics ----------
    try:
        preds = np.array(ds["predictions"])
        gts = np.array(ds["ground_truth"])
        seqs = np.array(ds["sequences"])

        test_acc = (preds == gts).mean()

        # re-compute CoWA locally ---------------------------------
        def count_color_variety(sequence: str) -> int:
            return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))

        def count_shape_variety(sequence: str) -> int:
            return len(set(tok[0] for tok in sequence.strip().split() if tok))

        def complexity_weight(seq: str) -> int:
            return count_color_variety(seq) + count_shape_variety(seq)

        weights = np.array([complexity_weight(s) for s in seqs])
        cowa = (weights * (preds == gts)).sum() / weights.sum()

        print(f"Test Accuracy: {test_acc:.3f} | Test CoWA: {cowa:.3f}")
    except Exception as e:
        print(f"Error computing evaluation metrics: {e}")
