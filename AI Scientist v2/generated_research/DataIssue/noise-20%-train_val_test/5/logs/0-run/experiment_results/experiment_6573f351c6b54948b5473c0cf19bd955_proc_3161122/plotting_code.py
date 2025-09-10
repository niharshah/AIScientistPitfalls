import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- load data -------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["optimizer_weight_decay"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = {}

# Identify run keys
run_keys = [k for k in spr_data.keys() if k.startswith("wd_")]
run_keys.sort(key=lambda x: float(x.split("_")[1]))

# -------------------- Accuracy curves -------------------- #
try:
    plt.figure(figsize=(6, 4))
    for rk in run_keys:
        epochs = np.arange(1, len(spr_data[rk]["metrics"]["train_acc"]) + 1)
        plt.plot(
            epochs, spr_data[rk]["metrics"]["train_acc"], "--", label=f"{rk} train"
        )
        plt.plot(epochs, spr_data[rk]["metrics"]["val_acc"], "-", label=f"{rk} val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Training vs Validation Accuracy")
    plt.legend(fontsize=6, ncol=2)
    fname = os.path.join(working_dir, "SPR_BENCH_acc_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# -------------------- Loss curves -------------------- #
try:
    plt.figure(figsize=(6, 4))
    for rk in run_keys:
        epochs = np.arange(1, len(spr_data[rk]["metrics"]["train_loss"]) + 1)
        plt.plot(
            epochs, spr_data[rk]["metrics"]["train_loss"], "--", label=f"{rk} train"
        )
        plt.plot(epochs, spr_data[rk]["metrics"]["val_loss"], "-", label=f"{rk} val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend(fontsize=6, ncol=2)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# -------------------- Test accuracy bar chart -------------------- #
try:
    plt.figure(figsize=(5, 3))
    test_accs = [spr_data[rk]["test_acc"] for rk in run_keys]
    plt.bar(range(len(run_keys)), test_accs, tick_label=[rk for rk in run_keys])
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1)
    plt.title("SPR_BENCH: Test Accuracy by Weight Decay")
    for i, v in enumerate(test_accs):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fname = os.path.join(working_dir, "SPR_BENCH_test_acc_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating test-acc bar plot: {e}")
    plt.close()

# -------------------- Confusion matrix for best run -------------------- #
try:
    best = spr_data["best_run"]
    preds = np.array(best["predictions"])
    gts = np.array(best["ground_truth"])
    labels = sorted(list(set(gts)))
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for p, g in zip(preds, gts):
        cm[g, p] += 1
    plt.figure(figsize=(4, 4))
    im = plt.imshow(cm, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SPR_BENCH Best Run Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(labels)
    plt.yticks(labels)
    fname = os.path.join(working_dir, "SPR_BENCH_best_confmat.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# -------------------- Print evaluation metrics -------------------- #
if spr_data:
    print("Test accuracies by weight_decay:")
    for rk, acc in zip(run_keys, test_accs):
        print(f"{rk}: {acc:.4f}")
    print(f"Best validation accuracy: {best['val_acc']:.4f}")
