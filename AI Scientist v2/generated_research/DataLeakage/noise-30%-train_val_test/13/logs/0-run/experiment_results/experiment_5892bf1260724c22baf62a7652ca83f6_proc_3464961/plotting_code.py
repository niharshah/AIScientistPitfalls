import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_exp = experiment_data["dropout_rate"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_exp = {}

if spr_exp:  # proceed only if data exists
    dropouts = sorted(spr_exp.keys(), key=lambda x: float(x.split("_")[1]))
    epochs = spr_exp[dropouts[0]]["epochs"]

    # ---------- 1. F1 curves ----------
    try:
        plt.figure()
        for dr in dropouts:
            plt.plot(epochs, spr_exp[dr]["metrics"]["train_f1"], label=f"train {dr}")
            plt.plot(epochs, spr_exp[dr]["metrics"]["val_f1"], "--", label=f"val {dr}")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH: Macro-F1 vs Epochs (dropout sweep)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_f1_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curves: {e}")
        plt.close()

    # ---------- 2. Loss curves ----------
    try:
        plt.figure()
        for dr in dropouts:
            plt.plot(epochs, spr_exp[dr]["losses"]["train"], label=f"train {dr}")
            plt.plot(epochs, spr_exp[dr]["losses"]["val"], "--", label=f"val {dr}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Loss vs Epochs (dropout sweep)")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ---------- 3. Test F1 bar ----------
    try:
        plt.figure()
        test_f1s = [spr_exp[dr]["test_f1"] for dr in dropouts]
        plt.bar(dropouts, test_f1s, color="skyblue")
        plt.ylabel("Test Macro-F1")
        plt.title("SPR_BENCH: Test F1 by Dropout")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_testF1_bar.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test F1 bar: {e}")
        plt.close()

    # ---------- 4. Confusion matrix for best dropout ----------
    try:
        best_idx = int(np.argmax(test_f1s))
        best_dr = dropouts[best_idx]
        y_true = spr_exp[best_dr]["ground_truth"]
        y_pred = spr_exp[best_dr]["predictions"]
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.title(f"SPR_BENCH Confusion Matrix (best {best_dr})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.savefig(os.path.join(working_dir, f"SPR_BENCH_confusion_{best_dr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- summary print ----------
    print("Test macro-F1 by dropout:")
    for dr, f1 in zip(dropouts, test_f1s):
        print(f"{dr}: {f1:.4f}")
