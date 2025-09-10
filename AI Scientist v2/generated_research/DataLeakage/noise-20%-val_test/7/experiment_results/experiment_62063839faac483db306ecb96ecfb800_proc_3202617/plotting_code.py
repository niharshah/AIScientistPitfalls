import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data_key = experiment_data.get("hidden_dim", {}).get("SPR_BENCH", {})
metrics_all = data_key.get("metrics", {})
losses_all = data_key.get("losses", {})
preds = np.array(data_key.get("predictions", []))
gts = np.array(data_key.get("ground_truth", []))
rule_preds = np.array(data_key.get("rule_preds", []))
hidden_dims = [hd for hd in metrics_all if isinstance(hd, int)]
hidden_dims.sort()

# 1) Accuracy curves
try:
    plt.figure(figsize=(6, 4))
    for hd in hidden_dims:
        epochs = np.arange(1, len(metrics_all[hd]["train_acc"]) + 1)
        plt.plot(epochs, metrics_all[hd]["train_acc"], label=f"{hd}-train")
        plt.plot(epochs, metrics_all[hd]["val_acc"], linestyle="--", label=f"{hd}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SPR_BENCH: Training vs Validation Accuracy")
    plt.legend(fontsize=7, ncol=2)
    fname = os.path.join(working_dir, "SPR_BENCH_acc_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curves: {e}")
    plt.close()

# 2) Loss curves
try:
    plt.figure(figsize=(6, 4))
    for hd in hidden_dims:
        epochs = np.arange(1, len(metrics_all[hd]["val_loss"]) + 1)
        plt.plot(epochs, losses_all[hd]["train"], label=f"{hd}-train")
        plt.plot(epochs, metrics_all[hd]["val_loss"], linestyle="--", label=f"{hd}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend(fontsize=7, ncol=2)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 3) Final validation accuracy per hidden dim
try:
    plt.figure(figsize=(5, 3))
    final_val_acc = [metrics_all[hd]["val_acc"][-1] for hd in hidden_dims]
    plt.bar([str(hd) for hd in hidden_dims], final_val_acc, color="skyblue")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Final Val Accuracy")
    plt.title("SPR_BENCH: Final Validation Accuracy per Hidden Size")
    fname = os.path.join(working_dir, "SPR_BENCH_val_acc_bar.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"Error creating val accuracy bar: {e}")
    plt.close()

# 4) Confusion matrix
try:
    if preds.size and gts.size:
        classes = sorted(np.unique(np.concatenate([gts, preds])))
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[gt, pr] += 1
        plt.figure(figsize=(4, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046)
        plt.xticks(range(len(classes)), classes)
        plt.yticks(range(len(classes)), classes)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH: Confusion Matrix (Test)")
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8
                )
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# 5) Accuracy vs Fidelity
try:
    best_test_acc = metrics_all.get(
        "best_test_acc", data_key.get("metrics", {}).get("best_test_acc")
    )
    best_fidelity = metrics_all.get(
        "best_fidelity", data_key.get("metrics", {}).get("best_fidelity")
    )
    if best_test_acc is None:
        best_test_acc = data_key.get("metrics", {}).get("best_test_acc")
    if best_fidelity is None:
        best_fidelity = data_key.get("metrics", {}).get("best_fidelity")
    vals = [best_test_acc, best_fidelity] if best_test_acc is not None else []
    if vals:
        plt.figure(figsize=(4, 3))
        plt.bar(["Test Acc", "Rule Fidelity"], vals, color=["green", "orange"])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Test Accuracy vs Rule Fidelity")
        fname = os.path.join(working_dir, "SPR_BENCH_acc_vs_fidelity.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Error creating acc vs fidelity bar: {e}")
    plt.close()

# Print metrics
best_test_acc = data_key.get("metrics", {}).get("best_test_acc", None)
best_fidelity = data_key.get("metrics", {}).get("best_fidelity", None)
if best_test_acc is not None and best_fidelity is not None:
    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    print(f"Rule Fidelity: {best_fidelity:.4f}")
