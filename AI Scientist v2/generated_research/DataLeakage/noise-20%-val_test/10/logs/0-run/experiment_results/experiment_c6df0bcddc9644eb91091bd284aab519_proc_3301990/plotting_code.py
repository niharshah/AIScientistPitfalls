import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import itertools

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
    spr_dict = experiment_data.get("embedding_dim", {}).get("SPR_BENCH", {})
    if not spr_dict:
        print("No SPR_BENCH data found.")
    else:
        # Collect stats
        dims = sorted([int(k.split("_")[-1]) for k in spr_dict.keys()])
        losses_tr, losses_val, f1s_tr, f1s_val = {}, {}, {}, {}
        for dim in dims:
            rec = spr_dict[f"emb_{dim}"]
            losses_tr[dim] = rec["losses"]["train"]
            losses_val[dim] = rec["losses"]["val"]
            f1s_tr[dim] = rec["metrics"]["train_f1"]
            f1s_val[dim] = rec["metrics"]["val_f1"]

        # ---------- Plot 1: Loss curves ----------
        try:
            plt.figure()
            for dim in dims:
                plt.plot(losses_tr[dim], "--", label=f"train (emb={dim})")
                plt.plot(losses_val[dim], label=f"val   (emb={dim})")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR_BENCH: Training/Validation Loss vs Epoch")
            plt.legend()
            fname = os.path.join(working_dir, "spr_loss_curves.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating loss curve plot: {e}")
            plt.close()

        # ---------- Plot 2: F1 curves ----------
        try:
            plt.figure()
            for dim in dims:
                plt.plot(f1s_tr[dim], "--", label=f"train (emb={dim})")
                plt.plot(f1s_val[dim], label=f"val   (emb={dim})")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title("SPR_BENCH: Training/Validation Macro-F1 vs Epoch")
            plt.legend()
            fname = os.path.join(working_dir, "spr_f1_curves.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating F1 curve plot: {e}")
            plt.close()

        # ---------- Plot 3: Confusion matrix for best embedding dim ----------
        try:
            # choose best dim by highest test_f1
            best_dim = max(dims, key=lambda d: spr_dict[f"emb_{d}"]["test_f1"])
            best_rec = spr_dict[f"emb_{best_dim}"]
            preds = np.array(best_rec["predictions"])
            gts = np.array(best_rec["ground_truth"])
            cm = confusion_matrix(gts, preds)
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title(f"SPR_BENCH Confusion Matrix (emb={best_dim})")
            plt.colorbar()
            tick_marks = np.arange(cm.shape[0])
            plt.xticks(tick_marks, tick_marks)
            plt.yticks(tick_marks, tick_marks)
            thresh = cm.max() / 2.0
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(
                    j,
                    i,
                    cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            fname = os.path.join(working_dir, f"spr_confusion_matrix_emb{best_dim}.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Error creating confusion matrix plot: {e}")
            plt.close()
