import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    runs = experiment_data["learning_rate"]["SPR_BENCH"]["runs"]
    lrs = [r["lr_ft"] for r in runs]

    # ---------- 1. Fine-tune loss curves ----------
    try:
        plt.figure()
        for r, lr in zip(runs, lrs):
            plt.plot(r["losses"]["train"], label=f"{lr:.0e} train")
            plt.plot(r["losses"]["val"], label=f"{lr:.0e} val", ls="--")
        plt.title("SPR_BENCH: Fine-tune Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_finetune_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot finetune loss: {e}")
        plt.close()

    # ---------- 2. Pre-train loss curves ----------
    try:
        plt.figure()
        for r, lr in zip(runs, lrs):
            plt.plot(r["losses"]["pretrain"], label=f"{lr:.0e}")
        plt.title("SPR_BENCH: Pre-train Contrastive Loss")
        plt.xlabel("Epoch")
        plt.ylabel("NT-Xent Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_pretrain_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot pretrain loss: {e}")
        plt.close()

    # ---------- 3. SCHM curves ----------
    try:
        plt.figure()
        for r, lr in zip(runs, lrs):
            plt.plot(r["metrics"]["SCHM"], label=f"{lr:.0e}")
        plt.title("SPR_BENCH: SCHM Metric over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("SCHM")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_SCHM_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot SCHM curves: {e}")
        plt.close()

    # ---------- 4. Final SCHM vs LR ----------
    try:
        final_schm = [r["metrics"]["SCHM"][-1] for r in runs]
        plt.figure()
        plt.bar([f"{lr:.0e}" for lr in lrs], final_schm, color="skyblue")
        plt.title("SPR_BENCH: Final-Epoch SCHM vs Learning-Rate")
        plt.xlabel("Fine-tune Learning-Rate")
        plt.ylabel("SCHM")
        fname = os.path.join(working_dir, "SPR_BENCH_final_SCHM_vs_lr.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot SCHM vs lr: {e}")
        plt.close()

    # ---------- 5. Correct / Incorrect for best LR ----------
    try:
        best_idx = int(np.argmax([r["metrics"]["SCHM"][-1] for r in runs]))
        best_run = runs[best_idx]
        preds = np.array(best_run["predictions"])
        truth = np.array(best_run["ground_truth"])
        correct = (preds == truth).sum()
        incorrect = len(preds) - correct
        plt.figure()
        plt.bar(["Correct", "Incorrect"], [correct, incorrect], color=["green", "red"])
        plt.title(f"SPR_BENCH: Prediction Quality (LR={lrs[best_idx]:.0e})")
        fname = os.path.join(
            working_dir, f"SPR_BENCH_correct_vs_incorrect_lr_{lrs[best_idx]:.0e}.png"
        )
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating plot correctness: {e}")
        plt.close()
