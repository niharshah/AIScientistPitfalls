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


# ---------- helper for CoWA ----------
def colour_of(token: str) -> str:
    return token[1:] if len(token) > 1 else ""


def shape_of(token: str) -> str:
    return token[0]


def complexity_weight(seq: str) -> int:
    return len({colour_of(t) for t in seq.split() if t}) + len(
        {shape_of(t) for t in seq.split() if t}
    )


# ---------- per-dataset plots ----------
datasets = list(experiment_data.keys()) if experiment_data else []
for ds_name in datasets:
    ds = experiment_data[ds_name]
    # ---- extract series ----
    epochs = np.arange(1, len(ds["losses"]["train"]) + 1)
    train_loss = ds["losses"]["train"]
    val_loss = ds["losses"]["val"]
    train_acc = [m["acc"] for m in ds["metrics"]["train"]]
    val_acc = [m["acc"] for m in ds["metrics"]["val"]]
    val_cwa = [
        m["CompWA"] if "CompWA" in m else m.get("cowa", np.nan)
        for m in ds["metrics"]["val"]
    ]

    # 1) Loss curve ----------------------------------------------------
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

    # 2) Accuracy curve ------------------------------------------------
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

    # 3) CoWA curve ----------------------------------------------------
    try:
        plt.figure()
        plt.plot(epochs, val_cwa, label="Validation CoWA")
        plt.xlabel("Epoch")
        plt.ylabel("Complexity-Weighted Accuracy")
        plt.title(f"{ds_name} CoWA Curve\nValidation Set Only")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{ds_name}_cowa_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating CoWA plot for {ds_name}: {e}")
        plt.close()

    # ---- evaluation metrics ----
    try:
        preds = np.array(ds["predictions"])
        gts = np.array(ds["ground_truth"])
        seqs = np.array(ds["sequences"])

        test_acc = (preds == gts).mean()
        weights = np.array([complexity_weight(s) for s in seqs])
        cowa = (weights * (preds == gts)).sum() / weights.sum()
        print(f"{ds_name} – Test Accuracy: {test_acc:.3f} | Test CoWA: {cowa:.3f}")
    except Exception as e:
        print(f"Error computing metrics for {ds_name}: {e}")

# ---------- comparison plots across datasets (if >1) ----------
if len(datasets) > 1:
    # Validation loss comparison
    try:
        plt.figure()
        for ds_name in datasets:
            val_loss = experiment_data[ds_name]["losses"]["val"]
            plt.plot(np.arange(1, len(val_loss) + 1), val_loss, label=f"{ds_name} Val")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss Comparison Across Datasets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_val_loss.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison loss plot: {e}")
        plt.close()

    # Validation accuracy comparison
    try:
        plt.figure()
        for ds_name in datasets:
            val_acc = [m["acc"] for m in experiment_data[ds_name]["metrics"]["val"]]
            plt.plot(np.arange(1, len(val_acc) + 1), val_acc, label=f"{ds_name} Val")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy Comparison Across Datasets")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "comparison_val_accuracy.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating comparison accuracy plot: {e}")
        plt.close()
