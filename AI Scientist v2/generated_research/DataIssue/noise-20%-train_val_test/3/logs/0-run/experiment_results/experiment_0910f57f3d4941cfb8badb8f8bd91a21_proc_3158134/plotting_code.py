import matplotlib.pyplot as plt
import numpy as np
import os

# -------- paths & loading --------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp = experiment_data.get("dropout_tuning", {}).get("SPR_BENCH", {})
by_dropout = exp.get("by_dropout", {})

# -------- training curves per dropout (≤4 figs) --------
for i, (dr, info) in enumerate(sorted(by_dropout.items())[:5]):  # safety slice
    try:
        epochs = info["epochs"]
        tr_loss = info["train_loss"]
        val_loss = info["val_loss"]
        val_f1 = info["val_f1"]

        fig, ax1 = plt.subplots()
        ax1.plot(epochs, tr_loss, label="Train Loss", color="tab:blue")
        ax1.plot(epochs, val_loss, label="Val Loss", color="tab:orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_f1, label="Val F1", color="tab:green")
        ax2.set_ylabel("Macro F1")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")

        plt.title(f"SPR_BENCH Training Curves (dropout={dr})")
        fname = os.path.join(working_dir, f"SPR_BENCH_dropout_{dr}_training_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot for dropout {dr}: {e}")
        plt.close()

# -------- label distribution bar chart (1 fig) --------
try:
    preds = np.array(exp.get("predictions", []))
    gts = np.array(exp.get("ground_truth", []))
    if preds.size and gts.size:
        num_classes = max(preds.max(), gts.max()) + 1
        gt_cnt = np.bincount(gts, minlength=num_classes)
        pr_cnt = np.bincount(preds, minlength=num_classes)

        idx = np.arange(num_classes)
        width = 0.35
        plt.figure()
        plt.bar(idx - width / 2, gt_cnt, width, label="Ground Truth")
        plt.bar(idx + width / 2, pr_cnt, width, label="Predictions")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title(
            "SPR_BENCH Class Distribution\nLeft: Ground Truth, Right: Generated Samples"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_class_distribution.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating class distribution plot: {e}")
    plt.close()
