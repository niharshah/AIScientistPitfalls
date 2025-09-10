import matplotlib.pyplot as plt
import numpy as np
import os

# ---- paths & load ----
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

models = list(experiment_data.keys())
epochs = np.arange(1, len(experiment_data[models[0]]["losses"]["train"]) + 1)

# ---- helper for colors ----
colors = {"baseline": "tab:blue", "symbolic": "tab:orange"}

# 1: training loss
try:
    plt.figure()
    for m in models:
        plt.plot(
            epochs, experiment_data[m]["losses"]["train"], label=m, color=colors.get(m)
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Training Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_train_loss_baseline_vs_symbolic.png"))
    plt.close()
except Exception as e:
    print(f"Error training-loss plot: {e}")
    plt.close()

# 2: validation loss
try:
    plt.figure()
    for m in models:
        plt.plot(
            epochs, experiment_data[m]["losses"]["val"], label=m, color=colors.get(m)
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_val_loss_baseline_vs_symbolic.png"))
    plt.close()
except Exception as e:
    print(f"Error val-loss plot: {e}")
    plt.close()

# 3: training macro-F1
try:
    plt.figure()
    for m in models:
        plt.plot(
            epochs, experiment_data[m]["metrics"]["train"], label=m, color=colors.get(m)
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Training Macro-F1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_train_f1_baseline_vs_symbolic.png"))
    plt.close()
except Exception as e:
    print(f"Error train-F1 plot: {e}")
    plt.close()

# 4: validation macro-F1
try:
    plt.figure()
    for m in models:
        plt.plot(
            epochs, experiment_data[m]["metrics"]["val"], label=m, color=colors.get(m)
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Validation Macro-F1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_val_f1_baseline_vs_symbolic.png"))
    plt.close()
except Exception as e:
    print(f"Error val-F1 plot: {e}")
    plt.close()


# 5: confusion matrices
def conf_mat(preds, gts, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1
    return cm


try:
    n_cls = len(set(experiment_data[models[0]]["ground_truth"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m in zip(axes, models):
        cm = conf_mat(
            experiment_data[m]["predictions"], experiment_data[m]["ground_truth"], n_cls
        )
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{m.capitalize()} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(
                    j, i, cm[i, j], ha="center", va="center", fontsize=6, color="black"
                )
    fig.suptitle("SPR_BENCH-dev: Left Baseline, Right Symbolic")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "spr_confusion_matrices_baseline_vs_symbolic.png")
    )
    plt.close()
except Exception as e:
    print(f"Error confusion-matrix plot: {e}")
    plt.close()

# ---- numeric summary ----
for m in models:
    best_f1 = np.max(experiment_data[m]["metrics"]["val"])
    final_f1 = experiment_data[m]["metrics"]["val"][-1]
    print(
        f"{m.capitalize():8}: best val F1 = {best_f1:.4f} | final val F1 = {final_f1:.4f}"
    )
