import matplotlib.pyplot as plt
import numpy as np
import os

# working directory for plots
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------------------------------------------------- #
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# --------------------------------------------------------------------- #
# extract SPR_BENCH results
bench = experiment_data.get("dropout_tuning", {}).get("SPR_BENCH", {})
if not bench:
    print("No SPR_BENCH data found.")
    exit()

dropouts = sorted(bench.keys(), key=float)
epochs = len(bench[dropouts[0]]["metrics"]["train_acc"])

# gather stats
test_accs = {dp: bench[dp]["test_acc"] for dp in dropouts}
best_dp = max(test_accs, key=test_accs.get)
best_acc = test_accs[best_dp]

print("=== Test Accuracies ===")
for dp, acc in test_accs.items():
    print(f"Dropout {dp}: {acc*100:.2f}%")
print(f"\nBest dropout: {best_dp} (test acc {best_acc*100:.2f}%)")

# --------------------------------------------------------------------- #
# 1) accuracy curves
try:
    plt.figure()
    for dp in dropouts:
        tr = bench[dp]["metrics"]["train_acc"]
        val = bench[dp]["metrics"]["val_acc"]
        plt.plot(range(1, epochs + 1), tr, label=f"dropout {dp} train", linestyle="-")
        plt.plot(range(1, epochs + 1), val, label=f"dropout {dp} val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Training and Validation Accuracy vs Epoch\n(Dropout tuning)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# --------------------------------------------------------------------- #
# 2) loss curves
try:
    plt.figure()
    for dp in dropouts:
        tr = bench[dp]["losses"]["train_loss"]
        val = bench[dp]["losses"]["val_loss"]
        plt.plot(range(1, epochs + 1), tr, label=f"dropout {dp} train", linestyle="-")
        plt.plot(range(1, epochs + 1), val, label=f"dropout {dp} val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training and Validation Loss vs Epoch\n(Dropout tuning)")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# --------------------------------------------------------------------- #
# 3) bar chart of test accuracies
try:
    plt.figure()
    x = np.arange(len(dropouts))
    y = [test_accs[dp] * 100 for dp in dropouts]
    plt.bar(x, y, color="skyblue")
    plt.xticks(x, dropouts)
    plt.ylabel("Test Accuracy (%)")
    plt.title("SPR_BENCH: Test Accuracy by Dropout Rate")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar chart: {e}")
    plt.close()

# --------------------------------------------------------------------- #
# 4) confusion matrix for best dropout
try:
    preds = np.array(bench[best_dp]["predictions"])
    gts = np.array(bench[best_dp]["ground_truth"])
    labels = np.unique(gts)
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(gts, preds):
        cm[t, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"SPR_BENCH: Confusion Matrix (Best dropout={best_dp})")
    plt.xticks(labels)
    plt.yticks(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
            )
    fname = os.path.join(
        working_dir, f"SPR_BENCH_confusion_matrix_best_dropout_{best_dp}.png"
    )
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
