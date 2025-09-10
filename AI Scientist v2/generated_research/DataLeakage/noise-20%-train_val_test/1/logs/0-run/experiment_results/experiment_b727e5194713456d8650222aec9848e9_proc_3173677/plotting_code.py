import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
exp_file_cands = [
    "experiment_data.npy",
    os.path.join(working_dir, "experiment_data.npy"),
]
exp_data = None
for f in exp_file_cands:
    if os.path.exists(f):
        try:
            exp_data = np.load(f, allow_pickle=True).item()
            break
        except Exception as e:
            print(f"Error loading {f}: {e}")
if exp_data is None:
    print("experiment_data.npy not found, aborting plotting.")
    quit()

# Safely navigate the expected keys
try:
    run = exp_data["no_label_smoothing"]["SPR_BENCH"]
except KeyError as e:
    print(f"Expected keys missing: {e}")
    quit()

epochs = range(1, len(run["losses"]["train"]) + 1)

# 1) Loss curve
try:
    plt.figure()
    plt.plot(epochs, run["losses"]["train"], label="Train")
    plt.plot(epochs, run["losses"]["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Loss Curve\nTrain vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# 2) Accuracy curve
try:
    plt.figure()
    plt.plot(epochs, run["metrics"]["train_acc"], label="Train")
    plt.plot(epochs, run["metrics"]["val_acc"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH Accuracy Curve\nTrain vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# 3) Macro-F1 curve
try:
    plt.figure()
    plt.plot(epochs, run["metrics"]["train_f1"], label="Train")
    plt.plot(epochs, run["metrics"]["val_f1"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH Macro-F1 Curve\nTrain vs Validation")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve: {e}")
    plt.close()

# 4) Confusion matrix (test set)
try:
    preds = np.array(run["predictions"])
    gts = np.array(run["ground_truth"])
    num_cls = int(max(preds.max(), gts.max())) + 1
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("SPR_BENCH Confusion Matrix\nLeft: GT rows, Right: Pred cols")
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# Print final evaluation metrics
try:
    test_acc = (preds == gts).mean()
    # macro-F1
    f1s = []
    for c in range(num_cls):
        tp = ((preds == c) & (gts == c)).sum()
        fp = ((preds == c) & (gts != c)).sum()
        fn = ((preds != c) & (gts == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    test_f1 = float(np.mean(f1s))
    print(f"Test Accuracy: {test_acc*100:.2f}%  |  Test Macro-F1: {test_f1:.4f}")
except Exception as e:
    print(f"Error computing final metrics: {e}")
