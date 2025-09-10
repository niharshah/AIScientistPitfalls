import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# Load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp_dict = experiment_data["batch_size"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp_dict = {}

# Skip plotting if nothing is loaded
if exp_dict:
    tags = sorted(exp_dict.keys())  # e.g. ['SPR_BENCH_bs64', ...]
    epochs_sup = max(len(v["losses"]["train"]) for v in exp_dict.values())
    # --------------------------------------------------------
    # 1) Training-loss curves
    try:
        plt.figure()
        for tag in tags:
            losses = exp_dict[tag]["losses"]["train"]
            plt.plot(range(1, len(losses) + 1), losses, label=tag.split("_")[-1])
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Training Loss vs Epoch")
        plt.legend(title="Batch Size")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_training_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating training-loss plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # 2) Dev HSCA curves
    try:
        plt.figure()
        for tag in tags:
            hsca = exp_dict[tag]["metrics"]["train"]  # dev HSCA per epoch
            plt.plot(range(1, len(hsca) + 1), hsca, label=tag.split("_")[-1])
        plt.xlabel("Epoch")
        plt.ylabel("HSCA")
        plt.title("SPR_BENCH – Dev HSCA vs Epoch")
        plt.legend(title="Batch Size")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_dev_HSCA_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HSCA curve plot: {e}")
        plt.close()

    # --------------------------------------------------------
    # 3) Final test HSCA bar chart
    final_scores = {tag: exp_dict[tag]["metrics"]["val"][0] for tag in tags}
    try:
        plt.figure()
        xs = np.arange(len(tags))
        plt.bar(xs, [final_scores[t] for t in tags], color="skyblue")
        plt.xticks(xs, [t.split("_")[-1] for t in tags])
        plt.ylabel("HSCA")
        plt.title("SPR_BENCH – Final Test HSCA by Batch Size")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_final_test_HSCA.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating HSCA bar chart: {e}")
        plt.close()

    # --------------------------------------------------------
    # 4) Confusion matrix for best model
    try:
        best_tag = max(final_scores, key=final_scores.get)
        preds = np.array(exp_dict[best_tag]["predictions"])
        gts = np.array(exp_dict[best_tag]["ground_truth"])
        cm = np.zeros((2, 2), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1

        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title(f'SPR_BENCH – Confusion Matrix (Best {best_tag.split("_")[-1]})')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks([0, 1], ["Class 0", "Class 1"])
        plt.yticks([0, 1], ["Class 0", "Class 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        fname = os.path.join(
            working_dir, f'SPR_BENCH_confusion_best_{best_tag.split("_")[-1]}.png'
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # --------------------------------------------------------
    # Print final metrics
    print("Final Test HSCA by batch size:")
    for t, v in final_scores.items():
        print(f"{t}: {v:.4f}")
else:
    print("No experiment data available – nothing to plot.")
