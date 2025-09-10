import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

tags = list(experiment_data.keys())
test_mcc_scores = []

for tag in tags:
    d = experiment_data[tag]["SPR"]
    epochs = d["epochs"]
    # -------- Loss curve --------
    try:
        plt.figure()
        plt.plot(epochs, d["losses"]["train"], label="train")
        plt.plot(epochs, d["losses"]["val"], label="val")
        plt.title(f"SPR Loss Curve ({tag})")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.legend()
        fname = f"SPR_loss_curve_{tag}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {tag}: {e}")
        plt.close()

    # -------- MCC curve --------
    try:
        plt.figure()
        plt.plot(epochs, d["metrics"]["train_MCC"], label="train")
        plt.plot(epochs, d["metrics"]["val_MCC"], label="val")
        plt.title(f"SPR MCC Curve ({tag})")
        plt.xlabel("Epoch")
        plt.ylabel("MCC")
        plt.legend()
        fname = f"SPR_MCC_curve_{tag}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating MCC curve for {tag}: {e}")
        plt.close()

    # -------- Confusion matrix --------
    try:
        preds = np.array(d["predictions"])
        gts = np.array(d["ground_truth"])
        cm = np.zeros((2, 2), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[int(gt), int(pr)] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.colorbar()
        plt.title(
            f"SPR Confusion Matrix ({tag})\nLeft: Ground Truth, Right: Predictions"
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0, 1], [0, 1])
        plt.yticks([0, 1], [0, 1])
        fname = f"SPR_confusion_matrix_{tag}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {tag}: {e}")
        plt.close()

    # collect test MCC for comparison bar plot
    test_mcc_scores.append(d.get("test_MCC", np.nan))
    print(f"{tag}: Test MCC={d.get('test_MCC'):.3f}, Test F1={d.get('test_F1'):.3f}")

# -------- bar chart comparing test MCC --------
try:
    plt.figure()
    plt.bar(tags, test_mcc_scores, color=["tab:blue", "tab:orange"])
    plt.title(
        "SPR Test MCC Comparison\nLeft: with_positional_encoding, Right: no_positional_encoding"
    )
    plt.ylabel("Test MCC")
    fname = "SPR_test_MCC_comparison.png"
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating MCC comparison bar chart: {e}")
    plt.close()
