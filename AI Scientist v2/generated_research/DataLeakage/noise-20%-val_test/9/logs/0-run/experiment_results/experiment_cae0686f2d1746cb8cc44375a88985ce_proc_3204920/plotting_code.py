import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---- load data ----
try:
    exp = np.load(
        os.path.join(working_dir, "..", "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = exp["BATCH_SIZE"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit(0)

bss = np.array(ed["batch_sizes"])
train_accs = np.array(ed["metrics"]["train_acc"])  # shape [n_bs, epochs]
val_accs = np.array(ed["metrics"]["val_acc"])
train_losses = np.array(ed["losses"]["train"])
val_losses = np.array(ed["losses"]["val"])
rba_vals = np.array(ed["RBA_val"])
test_accs = np.array(ed["metrics"]["test_acc"])
epochs = np.arange(1, train_accs.shape[1] + 1)

saved_figs = []

# 1) accuracy curves
try:
    plt.figure()
    for i, bs in enumerate(bss):
        plt.plot(epochs, train_accs[i], label=f"Train bs={bs}")
        plt.plot(epochs, val_accs[i], "--", label=f"Val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH ‒ Training vs Validation Accuracy")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    saved_figs.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# 2) loss curves
try:
    plt.figure()
    for i, bs in enumerate(bss):
        plt.plot(epochs, train_losses[i], label=f"Train bs={bs}")
        plt.plot(epochs, val_losses[i], "--", label=f"Val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH ‒ Training vs Validation Loss")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    saved_figs.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) RBA curves
try:
    plt.figure()
    for i, bs in enumerate(bss):
        plt.plot(epochs, rba_vals[i], label=f"RBA Val bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Rule-based Accuracy")
    plt.title("SPR_BENCH ‒ RBA on Validation Set")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_RBA_curves.png")
    plt.savefig(fname)
    saved_figs.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating RBA plot: {e}")
    plt.close()

# 4) test accuracy vs batch size
try:
    plt.figure()
    plt.plot(bss, test_accs, marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Final Test Accuracy")
    plt.title("SPR_BENCH ‒ Test Accuracy vs Batch Size")
    plt.grid(True)
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_vs_bs.png")
    plt.savefig(fname)
    saved_figs.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test-accuracy plot: {e}")
    plt.close()

# 5) confusion matrix for best model
try:
    best_idx = np.argmax([max(v) for v in val_accs])
    preds = ed["predictions"][best_idx]
    gts = ed["ground_truth"][best_idx]
    num_cls = int(max(gts.max(), preds.max()) + 1)
    cm = np.zeros((num_cls, num_cls), dtype=int)
    for gt, pr in zip(gts, preds):
        cm[gt, pr] += 1
    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"SPR_BENCH ‒ Confusion Matrix (bs={bss[best_idx]})")
    fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix_best_bs.png")
    plt.savefig(fname)
    saved_figs.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

print("Saved figures:")
for f in saved_figs:
    print(" -", f)
