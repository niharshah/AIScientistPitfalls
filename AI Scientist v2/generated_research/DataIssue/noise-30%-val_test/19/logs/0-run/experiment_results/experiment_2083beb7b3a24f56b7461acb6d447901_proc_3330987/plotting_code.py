import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    hid_dict = experiment_data.get("hidden_size", {})
    best_info = hid_dict.get("best_model", {})
    best_hid = best_info.get("hidden_size")
    test_macro_f1 = best_info.get("test_macro_f1")

    print(f"Best hidden size: {best_hid}")
    print(f"Test Macro-F1: {test_macro_f1}")

    # collect list of hids (str) excluding the helper keys
    hids = [k for k in hid_dict.keys() if k not in ("best_model",)]

    # ---------- Plot 1: val loss per hid ----------
    try:
        plt.figure(figsize=(7, 4))
        for hid in hids:
            run = hid_dict[hid]
            plt.plot(run["epochs"], run["losses"]["val"], label=f"hid {hid}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("SPR_BENCH: Validation Loss over Epochs\nComparing Hidden Sizes")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_val_loss_by_hidden_size.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating val-loss plot: {e}")
        plt.close()

    # ---------- Plot 2: macro-F1 per hid ----------
    try:
        plt.figure(figsize=(7, 4))
        for hid in hids:
            run = hid_dict[hid]
            plt.plot(run["epochs"], run["metrics"]["val_macro_f1"], label=f"hid {hid}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Macro-F1")
        plt.title("SPR_BENCH: Validation Macro-F1 over Epochs\nComparing Hidden Sizes")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_macro_f1_by_hidden_size.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating macro-F1 plot: {e}")
        plt.close()

    # ---------- Plot 3: train vs val loss for best hid ----------
    try:
        if best_hid is not None:
            run = hid_dict[str(best_hid)]
            plt.figure(figsize=(7, 4))
            plt.plot(run["epochs"], run["losses"]["train"], label="Train")
            plt.plot(run["epochs"], run["losses"]["val"], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"SPR_BENCH: Train vs Val Loss\nBest Hidden Size = {best_hid}")
            plt.legend()
            plt.tight_layout()
            fname = os.path.join(
                working_dir, f"spr_bench_train_val_loss_best_hid_{best_hid}.png"
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating best-hid loss plot: {e}")
        plt.close()

    # ---------- Plot 4: confusion matrix ----------
    try:
        preds = best_info.get("predictions")
        gts = best_info.get("ground_truth")
        if preds is not None and gts is not None:
            cm = np.zeros((2, 2), dtype=int)
            for p, g in zip(preds, gts):
                cm[int(g), int(p)] += 1
            plt.figure(figsize=(4, 4))
            im = plt.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    plt.text(
                        j,
                        i,
                        cm[i, j],
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(
                "SPR_BENCH: Confusion Matrix\nLeft: Ground Truth, Right: Predictions"
            )
            plt.tight_layout()
            fname = os.path.join(
                working_dir, f"spr_bench_confusion_matrix_best_hid_{best_hid}.png"
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()
