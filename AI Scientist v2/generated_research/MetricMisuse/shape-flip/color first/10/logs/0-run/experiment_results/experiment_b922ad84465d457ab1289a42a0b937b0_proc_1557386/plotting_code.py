import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load data ---------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["NoGNN"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run, experiment_data = None, None

if run:
    losses = run["losses"]
    val_metrics = run["metrics"]["val"]
    test_metrics = run["metrics"]["test"]
    preds = run["predictions"]
    gtruth = run["ground_truth"]

    # Plot 1: training / validation loss
    try:
        plt.figure()
        epochs = np.arange(1, len(losses["train"]) + 1)
        plt.plot(epochs, losses["train"], label="Train Loss")
        plt.plot(epochs, losses["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # Plot 2: validation accuracies over epochs
    try:
        plt.figure()
        cwa = [d["cwa"] for d in val_metrics]
        swa = [d["swa"] for d in val_metrics]
        cpx = [d["cpxwa"] for d in val_metrics]
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, cpx, label="CpxWA")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted Accuracy")
        plt.title("SPR_BENCH Validation Accuracies")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_val_weighted_acc.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating val metric plot: {e}")
        plt.close()

    # Plot 3: final test metrics bar chart
    try:
        plt.figure()
        names, vals = zip(*test_metrics.items())
        plt.bar(names, vals)
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Test Weighted Accuracies")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_weighted_acc.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metric bar plot: {e}")
        plt.close()

    # Plot 4: confusion matrix heatmap
    try:
        import seaborn as sns  # lightweight extra; will fail gracefully if absent

        labels = sorted(set(gtruth))
        lab2id = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(gtruth, preds):
            cm[lab2id[t], lab2id[p]] += 1
        plt.figure()
        sns.heatmap(
            cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cbar=False
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH Confusion Matrix")
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # Plot 5: optional histogram of complexity weight vs correctness
    try:
        from collections import Counter

        seqs = experiment_data["NoGNN"]["SPR_BENCH"].get(
            "sequences_test", []
        )  # not stored by default
        if seqs:
            # recompute complexity weights like training script
            def cpx_w(s):
                return len({tok[1] for tok in s.split()}) + len(
                    {tok[0] for tok in s.split()}
                )

            weights = [cpx_w(s) for s in seqs]
            correct = [int(t == p) for t, p in zip(gtruth, preds)]
            plt.figure()
            plt.scatter(weights, correct, alpha=0.3)
            plt.yticks([0, 1], ["Wrong", "Correct"])
            plt.xlabel("Complexity Weight")
            plt.title("SPR_BENCH Correctness vs. Complexity")
            plt.savefig(
                os.path.join(working_dir, "SPR_BENCH_correct_vs_complexity.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating complexity scatter: {e}")
        plt.close()

    # -------- print final test metrics --------
    print("Final Test Metrics:", test_metrics)
