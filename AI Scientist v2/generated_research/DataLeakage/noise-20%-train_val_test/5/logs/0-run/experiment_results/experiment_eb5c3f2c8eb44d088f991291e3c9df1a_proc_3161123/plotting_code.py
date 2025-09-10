import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------- LOAD DATA -------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    spr_results = experiment_data["transformer_dropout_rate"]["SPR_BENCH"]
    dropout_keys = sorted(spr_results.keys(), key=lambda k: float(k.split("_")[-1]))

    # ------------- Helper to pick best model ------------- #
    best_key = max(
        dropout_keys,
        key=lambda k: (
            spr_results[k]["metrics"]["val"][-1]
            if spr_results[k]["metrics"]["val"]
            else -1
        ),
    )

    # ----------------- ACCURACY CURVES ------------------- #
    try:
        plt.figure()
        for k in dropout_keys:
            ep = range(1, len(spr_results[k]["metrics"]["train"]) + 1)
            plt.plot(ep, spr_results[k]["metrics"]["train"], label=f"train {k}")
            plt.plot(
                ep, spr_results[k]["metrics"]["val"], linestyle="--", label=f"val {k}"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            "SPR_BENCH: Training vs Validation Accuracy\n(Left: Train, Right: Val, Dropout Grid)"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # ------------------ LOSS CURVES ---------------------- #
    try:
        plt.figure()
        for k in dropout_keys:
            ep = range(1, len(spr_results[k]["losses"]["train"]) + 1)
            plt.plot(ep, spr_results[k]["losses"]["train"], label=f"train {k}")
            plt.plot(
                ep, spr_results[k]["losses"]["val"], linestyle="--", label=f"val {k}"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(
            "SPR_BENCH: Training vs Validation Loss\n(Left: Train, Right: Val, Dropout Grid)"
        )
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # ------------- TEST ACCURACY BAR CHART -------------- #
    try:
        plt.figure()
        test_accs = [
            (
                np.mean(
                    np.array(spr_results[k]["predictions"])
                    == np.array(spr_results[k]["ground_truth"])
                )
                if spr_results[k]["ground_truth"]
                else 0.0
            )
            for k in dropout_keys
        ]
        plt.bar([k.split("_")[-1] for k in dropout_keys], test_accs, color="skyblue")
        plt.ylim(0, 1)
        plt.xlabel("Dropout Rate")
        plt.ylabel("Test Accuracy")
        plt.title("SPR_BENCH: Test Accuracy per Dropout Rate")
        fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test accuracy bar chart: {e}")
        plt.close()

    # --------------- CONFUSION MATRIX ------------------- #
    try:
        preds = np.array(spr_results[best_key]["predictions"])
        gts = np.array(spr_results[best_key]["ground_truth"])
        if preds.size and gts.size:
            n_cls = len(set(gts))
            cm = np.zeros((n_cls, n_cls), dtype=int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                f'SPR_BENCH Confusion Matrix (Best Dropout={best_key.split("_")[-1]})'
            )
            for i in range(n_cls):
                for j in range(n_cls):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()
