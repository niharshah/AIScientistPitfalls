import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load experiment data ----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["learned_positional_embeddings"]["SPR_BENCH"]
except Exception as e:
    raise RuntimeError(f"Error loading experiment data: {e}")

loss_train, loss_val = np.asarray(exp["losses"]["train"]), np.asarray(
    exp["losses"]["val"]
)
acc_train, acc_val = np.asarray(exp["metrics"]["train_acc"]), np.asarray(
    exp["metrics"]["val_acc"]
)
f1_train, f1_val = np.asarray(exp["metrics"]["train_f1"]), np.asarray(
    exp["metrics"]["val_f1"]
)
preds, labels = np.asarray(exp["predictions"]), np.asarray(exp["ground_truth"])


# ---------------- utility for macro-F1 ----------------
def macro_f1(preds, labels, num_cls):
    f1s = []
    for c in range(num_cls):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


num_cls = int(labels.max()) + 1
test_acc = (preds == labels).mean()
test_f1 = macro_f1(preds, labels, num_cls)
print(f"SPR_BENCH Test accuracy: {test_acc*100:.2f}% | Test macroF1: {test_f1:.4f}")

# ---------------- plotting ----------------
# 1) Loss curves
try:
    plt.figure()
    epochs = np.arange(1, len(loss_train) + 1)
    plt.plot(epochs, loss_train, label="Train")
    plt.plot(epochs, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Accuracy curves
try:
    plt.figure()
    plt.plot(epochs, acc_train, label="Train")
    plt.plot(epochs, acc_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 3) F1 curves
try:
    plt.figure()
    plt.plot(epochs, f1_train, label="Train")
    plt.plot(epochs, f1_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Training vs Validation Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# 4) Confusion matrix
try:
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm_norm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR_BENCH Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
