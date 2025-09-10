import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["batch_size"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

batch_sizes = sorted(int(k.split("_")[-1]) for k in exp.keys())
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# 1) Loss curves
try:
    plt.figure()
    for c, bs in zip(colors, batch_sizes):
        logs = exp[f"bs_{bs}"]
        epochs = np.arange(1, len(logs["losses"]["train"]) + 1)
        plt.plot(
            epochs,
            logs["losses"]["train"],
            color=c,
            linestyle="-",
            label=f"train bs={bs}",
        )
        plt.plot(
            epochs, logs["losses"]["val"], color=c, linestyle="--", label=f"val bs={bs}"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) Macro-F1 curves
try:
    plt.figure()
    for c, bs in zip(colors, batch_sizes):
        logs = exp[f"bs_{bs}"]
        epochs = np.arange(1, len(logs["metrics"]["val"]) + 1)
        plt.plot(epochs, logs["metrics"]["val"], color=c, label=f"val F1 bs={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Validation Macro-F1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_macro_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# 3) Final-epoch F1 bar plot
try:
    final_f1 = [exp[f"bs_{bs}"]["metrics"]["val"][-1] for bs in batch_sizes]
    plt.figure()
    plt.bar([str(bs) for bs in batch_sizes], final_f1, color=colors[: len(batch_sizes)])
    plt.xlabel("Batch Size")
    plt.ylabel("Final Macro-F1")
    plt.title("SPR_BENCH: Final Macro-F1 vs Batch Size")
    for x, y in zip(batch_sizes, final_f1):
        plt.text(str(x), y + 0.01, f"{y:.2f}", ha="center", va="bottom")
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_final_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# 4) Confusion matrix for best batch size
try:
    best_idx = int(np.argmax(final_f1))
    best_bs = batch_sizes[best_idx]
    logs = exp[f"bs_{best_bs}"]
    preds = np.array(logs["predictions"])
    gts = np.array(logs["ground_truth"])
    cm = np.zeros((2, 2), dtype=int)
    for p, t in zip(preds, gts):
        cm[t, p] += 1
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"SPR_BENCH Confusion Matrix (bs={best_bs})")
    plt.colorbar()
    plt.savefig(
        os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_bs_{best_bs}.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
