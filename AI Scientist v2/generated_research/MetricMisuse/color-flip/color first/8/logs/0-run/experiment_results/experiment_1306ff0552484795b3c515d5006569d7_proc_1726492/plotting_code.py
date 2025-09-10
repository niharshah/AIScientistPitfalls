import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
# set / check working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ------------------------------------------------------------------
# helper: simple accuracy on stored predictions / gts
def simple_accuracy(preds, gts):
    preds, gts = np.asarray(preds), np.asarray(gts)
    return (preds == gts).mean() if preds.size else 0.0


# collect data
dropouts = sorted(
    experiment_data.get("dropout_rate", {}).keys(), key=lambda x: float(x)
)
train_loss = {}
val_loss = {}
final_test_acc = {}

for p in dropouts:
    exp = experiment_data["dropout_rate"][p]
    # losses come as list of (epoch, loss)
    train_loss[p] = np.array(exp["losses"]["train"])
    val_loss[p] = np.array(exp["losses"]["val"])
    final_test_acc[p] = simple_accuracy(exp["predictions"], exp["ground_truth"])

# ------------------------------------------------------------------
# Figure 1 : training / validation loss curves
try:
    plt.figure(figsize=(10, 4))
    # Left subplot: training loss
    plt.subplot(1, 2, 1)
    for p in dropouts:
        plt.plot(train_loss[p][:, 0], train_loss[p][:, 1], label=f"dropout {p}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Right subplot: validation loss
    plt.subplot(1, 2, 2)
    for p in dropouts:
        plt.plot(val_loss[p][:, 0], val_loss[p][:, 1], label=f"dropout {p}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    plt.suptitle("Left: Training Loss, Right: Validation Loss – SPR Synthetic")
    fname = os.path.join(working_dir, "spr_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve figure: {e}")
    plt.close()

# ------------------------------------------------------------------
# Figure 2 : final test accuracy per dropout
try:
    plt.figure()
    bars = [final_test_acc[p] for p in dropouts]
    plt.bar(range(len(dropouts)), bars, tick_label=dropouts)
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy vs Dropout – SPR Synthetic")
    fname = os.path.join(working_dir, "spr_test_accuracy.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy figure: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print evaluation metrics
print("Final test accuracy per dropout rate")
for p in dropouts:
    print(f"  dropout {p}: {final_test_acc[p]:.3f}")
