import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- working dir ----------
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

ds = "spr_bench"
if ds not in experiment_data:
    print(f'Dataset "{ds}" not found in experiment data.')
    exit()

loss_train = experiment_data[ds]["losses"]["train"]
loss_val = experiment_data[ds]["losses"]["val"]
bwa_train = experiment_data[ds]["metrics"]["train_BWA"]
bwa_val = experiment_data[ds]["metrics"]["val_BWA"]
y_pred = np.array(experiment_data[ds].get("predictions", []))
y_true = np.array(experiment_data[ds].get("ground_truth", []))
test_bwa = experiment_data[ds]["metrics"].get("test_BWA", None)

# ---------- 1) loss curves ----------
try:
    plt.figure()
    epochs = np.arange(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss — Dataset: SPR_BENCH")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ---------- 2) BWA curves ----------
try:
    plt.figure()
    epochs = np.arange(1, len(bwa_train) + 1)
    plt.plot(epochs, bwa_train, label="Train BWA")
    plt.plot(epochs, bwa_val, label="Validation BWA")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Weighted Accuracy")
    plt.title("Training vs Validation BWA — Dataset: SPR_BENCH")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_BWA_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating BWA curves: {e}")
    plt.close()

# ---------- 3) confusion matrix ----------
try:
    if y_true.size and y_pred.size:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = np.zeros((labels.size, labels.size), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[np.where(labels == t)[0][0], np.where(labels == p)[0][0]] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix — Dataset: SPR_BENCH")
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("Predictions or ground truth unavailable; skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------- print evaluation metric ----------
if test_bwa is not None:
    print(f"Test BWA: {test_bwa:.4f}")
