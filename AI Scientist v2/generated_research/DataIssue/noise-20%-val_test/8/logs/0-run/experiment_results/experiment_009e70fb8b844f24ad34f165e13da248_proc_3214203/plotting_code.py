import matplotlib.pyplot as plt
import numpy as np
import os

# ----------- paths / load -----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

# ----------- extract ---------------
bench = experiment_data["training_data_size_ablation"]["SPR_BENCH"]
fractions = np.array(bench["fractions"])
val_acc = np.array(bench["metrics"]["val_accuracy"])
test_acc = np.array(bench["metrics"]["test_accuracy"])
val_loss = np.array(bench["losses"]["val_logloss"])
y_true = np.array(bench["ground_truth"])
preds_dict = {float(k): np.array(v) for k, v in bench["predictions"].items()}

print("Fractions:", fractions)
print("Validation accuracy:", val_acc)
print("Test accuracy:", test_acc)
print("Validation log-loss:", val_loss)

# ----------- plot 1: accuracy curves -----------
try:
    plt.figure()
    plt.plot(fractions, val_acc, "o-", label="Validation accuracy")
    plt.plot(fractions, test_acc, "s-", label="Test accuracy")
    plt.xlabel("Training fraction")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Accuracy vs Training Data Size")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_vs_training_size.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ----------- plot 2: log-loss curve -----------
try:
    plt.figure()
    plt.plot(fractions, val_loss, "d-", color="purple")
    plt.xlabel("Training fraction")
    plt.ylabel("Validation Log-Loss")
    plt.title("SPR_BENCH: Validation Log-Loss vs Training Data Size")
    fname = os.path.join(working_dir, "SPR_BENCH_val_logloss_vs_training_size.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating log-loss plot: {e}")
    plt.close()

# ----------- plot 3: confusion matrix (100 % data) -----------
try:
    if 1.0 in preds_dict:
        y_pred = preds_dict[1.0]
        labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.title("SPR_BENCH Confusion Matrix (100% data)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.colorbar(im)
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix_100pct.png")
        plt.savefig(fname)
        plt.close()
    else:
        print("No predictions stored for fraction 1.0; skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
