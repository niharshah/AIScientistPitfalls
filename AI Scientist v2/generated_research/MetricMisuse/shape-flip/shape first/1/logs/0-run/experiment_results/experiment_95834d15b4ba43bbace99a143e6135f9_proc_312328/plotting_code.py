import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    dropout_dict = experiment_data.get("dropout_tuning", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    dropout_dict = {}


# Helper to sort keys numerically by dropout value
def _num(k):
    try:
        return float(k.split("_")[-1])
    except Exception:
        return 0.0


keys_sorted = sorted(dropout_dict.keys(), key=_num)

# ---------------------------------------------------------
# 1) Accuracy curves ----------------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    # Left: train acc
    plt.subplot(1, 2, 1)
    for k in keys_sorted:
        acc = dropout_dict[k]["metrics"].get("train_acc", [])
        plt.plot(range(1, len(acc) + 1), acc, label=k)
    plt.title("SPR_BENCH Train Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Right: val acc
    plt.subplot(1, 2, 2)
    for k in keys_sorted:
        acc = dropout_dict[k]["metrics"].get("val_acc", [])
        plt.plot(range(1, len(acc) + 1), acc, label=k)
    plt.title("SPR_BENCH Validation Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.suptitle("Left: Train Acc, Right: Val Acc (Dropout Sweep)")
    fname = os.path.join(working_dir, "SPR_BENCH_dropout_accuracy_curves.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------------------------------------------------------
# 2) Loss curves ---------------------------------------------------------------
try:
    plt.figure(figsize=(10, 4))
    # Left: train loss
    plt.subplot(1, 2, 1)
    for k in keys_sorted:
        loss = dropout_dict[k]["losses"].get("train", [])
        plt.plot(range(1, len(loss) + 1), loss, label=k)
    plt.title("SPR_BENCH Train Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Right: val loss
    plt.subplot(1, 2, 2)
    for k in keys_sorted:
        loss = dropout_dict[k]["losses"].get("val", [])
        plt.plot(range(1, len(loss) + 1), loss, label=k)
    plt.title("SPR_BENCH Validation Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.suptitle("Left: Train Loss, Right: Val Loss (Dropout Sweep)")
    fname = os.path.join(working_dir, "SPR_BENCH_dropout_loss_curves.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------------------------------------------------------
# 3) Final validation accuracy bar chart --------------------------------------
try:
    final_val_acc = [
        dropout_dict[k]["metrics"]["val_acc"][-1]
        for k in keys_sorted
        if dropout_dict[k]["metrics"]["val_acc"]
    ]
    labels = [k for k in keys_sorted if dropout_dict[k]["metrics"]["val_acc"]]
    plt.figure()
    plt.bar(labels, final_val_acc, color="skyblue")
    plt.ylabel("Validation Accuracy")
    plt.title("SPR_BENCH Final Validation Accuracy by Dropout")
    plt.xticks(rotation=45)
    fname = os.path.join(working_dir, "SPR_BENCH_final_val_accuracy_bar.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final val acc bar: {e}")
    plt.close()

# ---------------------------------------------------------
# 4) ZSRTA bar chart (if present) ---------------------------------------------
try:
    zs_scores = []
    zs_labels = []
    for k in keys_sorted:
        z = dropout_dict[k]["metrics"].get("ZSRTA", [])
        if z:  # ensure list not empty
            zs_scores.append(z[-1])
            zs_labels.append(k)
    if zs_scores:
        plt.figure()
        plt.bar(zs_labels, zs_scores, color="salmon")
        plt.ylabel("ZSRTA")
        plt.title("SPR_BENCH Zero-Shot Rule Transfer Accuracy")
        plt.xticks(rotation=45)
        fname = os.path.join(working_dir, "SPR_BENCH_zsrta_bar.png")
        plt.tight_layout()
        plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating ZSRTA bar: {e}")
    plt.close()
