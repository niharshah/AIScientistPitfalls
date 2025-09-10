import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# Helper to fetch
def get(entry):
    return experiment_data["bag_of_tokens"]["SPR_BENCH"][entry]


# 1) Loss curve ---------------------------------------------------------------
try:
    tr_loss = np.array(get("losses")["train"])
    val_loss = np.array(get("losses")["val"])
    plt.figure()
    plt.plot(tr_loss[:, 0], tr_loss[:, 1], label="Train")
    plt.plot(val_loss[:, 0], val_loss[:, 1], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        "SPR_BENCH Loss Curve\nLeft: Train, Right: Validation"
    )  # subtitle inside title line
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) SWA curve ---------------------------------------------------------------
try:
    tr_swa = np.array(get("metrics")["train"])
    val_swa = np.array(get("metrics")["val"])
    plt.figure()
    plt.plot(tr_swa[:, 0], tr_swa[:, 1], label="Train")
    plt.plot(val_swa[:, 0], val_swa[:, 1], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH SWA Curve\nLeft: Train, Right: Validation")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_SWA_curve.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve: {e}")
    plt.close()

# 3) Confusion matrix ---------------------------------------------------------
try:
    preds = np.array(get("predictions"))
    gts = np.array(get("ground_truth"))
    num_cls = int(max(preds.max(), gts.max())) + 1 if preds.size else 2
    conf = np.zeros((num_cls, num_cls), dtype=int)
    for t, p in zip(gts, preds):
        conf[t, p] += 1
    plt.figure()
    im = plt.imshow(conf, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR_BENCH Confusion Matrix\nLeft: Ground Truth, Right: Predicted")
    for i in range(num_cls):
        for j in range(num_cls):
            plt.text(j, i, conf[i, j], ha="center", va="center", color="black")
    save_path = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print("Finished plotting to", working_dir)
