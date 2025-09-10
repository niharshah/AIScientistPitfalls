import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------------------- load data ---------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    spr_data = experiment_data["learning_rate"]["SPR_BENCH"]
    train_losses = spr_data["losses"]["train"]
    val_losses = spr_data["losses"]["val"]
    train_metrics = spr_data["metrics"]["train"]
    val_metrics = spr_data["metrics"]["val"]
    lrs = spr_data["lr_used"]

    # regroup by lr
    def regroup(records, key):
        out = {}
        for d in records:
            lr = d["lr"]
            out.setdefault(lr, {"epoch": [], key: []})
            out[lr]["epoch"].append(d["epoch"])
            out[lr][key].append(d[key])
        return out

    tr_loss_by_lr = regroup(train_losses, "loss")
    va_loss_by_lr = regroup(val_losses, "loss")
    tr_f1_by_lr = regroup(train_metrics, "macro_f1")
    va_f1_by_lr = regroup(val_metrics, "macro_f1")

    # -------------------- 1) loss curves --------------------------
    try:
        plt.figure()
        for lr in lrs:
            plt.plot(
                tr_loss_by_lr[lr]["epoch"],
                tr_loss_by_lr[lr]["loss"],
                label=f"Train LR={lr}",
            )
            plt.plot(
                va_loss_by_lr[lr]["epoch"],
                va_loss_by_lr[lr]["loss"],
                linestyle="--",
                label=f"Val LR={lr}",
            )
        plt.title("SPR_BENCH: Train vs Val Loss across Learning Rates")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # -------------------- 2) macro-F1 curves ----------------------
    try:
        plt.figure()
        for lr in lrs:
            plt.plot(
                tr_f1_by_lr[lr]["epoch"],
                tr_f1_by_lr[lr]["macro_f1"],
                label=f"Train LR={lr}",
            )
            plt.plot(
                va_f1_by_lr[lr]["epoch"],
                va_f1_by_lr[lr]["macro_f1"],
                linestyle="--",
                label=f"Val LR={lr}",
            )
        plt.title("SPR_BENCH: Train vs Val Macro-F1 across Learning Rates")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 plot: {e}")
        plt.close()

    # -------- 3) final validation macro-F1 vs LR (bar chart) -----
    try:
        final_f1s = []
        for lr in lrs:
            # last entry corresponding to this lr is at index -1 of each 5-epoch block
            f1_vals = va_f1_by_lr[lr]["macro_f1"]
            final_f1s.append(f1_vals[-1])
        plt.figure()
        plt.bar([str(lr) for lr in lrs], final_f1s, color="skyblue")
        plt.title("SPR_BENCH: Final Validation Macro-F1 vs Learning Rate")
        plt.xlabel("Learning Rate")
        plt.ylabel("Macro-F1")
        fname = os.path.join(working_dir, "SPR_BENCH_valF1_vs_LR.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()

    # ---------------- 4) confusion matrix for best LR -------------
    try:
        best_lr = spr_data["best_lr"]
        preds_entry = next(d for d in spr_data["predictions"] if d["lr"] == best_lr)
        preds = np.array(preds_entry["values"])
        gts = np.array(spr_data["ground_truth"])
        labels = np.unique(gts)
        cm = np.zeros((labels.size, labels.size), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[gt, pr] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.title(f"SPR_BENCH Confusion Matrix (Best LR={best_lr})")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.xticks(labels)
        plt.yticks(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(
                    j, i, cm[i, j], ha="center", va="center", color="black", fontsize=7
                )
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix_bestLR.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------------------- print evaluation metric -----------------
    print(
        f"Best LR = {spr_data['best_lr']}  |  Dev Macro-F1 = {spr_data['best_val_f1']:.4f}"
    )
