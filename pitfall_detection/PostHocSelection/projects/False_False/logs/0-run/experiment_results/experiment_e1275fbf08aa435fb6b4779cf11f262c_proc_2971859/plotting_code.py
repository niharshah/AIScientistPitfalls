import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- paths --------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- data load ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["no_projection_head"]["SPR"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

# -------------------- figure 1: pretrain loss --------------------
try:
    plt.figure()
    pre_losses = ed["losses"]["pretrain"]
    plt.plot(range(1, len(pre_losses) + 1), pre_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.title("SPR Pre-training Loss (No Projection Head)")
    fname = os.path.join(working_dir, "SPR_pretrain_loss_no_projection_head.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating pretraining loss plot: {e}")
    plt.close()

# -------------------- figure 2: train vs val loss --------------------
try:
    plt.figure()
    train_losses = ed["losses"]["train"]
    val_losses = ed["losses"]["val"]
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker="o", label="Train")
    plt.plot(epochs, val_losses, marker="s", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR Training vs Validation Loss (No Projection Head)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_train_val_loss_no_projection_head.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating train/val loss plot: {e}")
    plt.close()

# -------------------- figure 3: validation metrics --------------------
try:
    plt.figure()
    swa = ed["metrics"]["val_SWA"]
    cwa = ed["metrics"]["val_CWA"]
    scwa = ed["metrics"]["val_SCWA"]
    epochs = range(1, len(swa) + 1)
    plt.plot(epochs, swa, marker="o", label="SWA")
    plt.plot(epochs, cwa, marker="s", label="CWA")
    plt.plot(epochs, scwa, marker="^", label="SCWA")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR Validation Metrics (No Projection Head)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_validation_metrics_no_projection_head.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# -------------------- figure 4: confusion heatmap --------------------
try:
    plt.figure()
    preds = np.array(ed["predictions"])
    trues = np.array(ed["ground_truth"])
    labels = np.unique(np.concatenate([preds, trues]))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(trues, preds):
        cm[t, p] += 1
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("SPR Confusion Matrix (No Projection Head)")
    fname = os.path.join(working_dir, "SPR_confusion_matrix_no_projection_head.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
