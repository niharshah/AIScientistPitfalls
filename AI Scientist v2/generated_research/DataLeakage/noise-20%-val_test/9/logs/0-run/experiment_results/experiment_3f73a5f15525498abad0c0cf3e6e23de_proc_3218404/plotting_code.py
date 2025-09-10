import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------ #
#                          LOAD EXPERIMENT                           #
# ------------------------------------------------------------------ #
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ------------------------------------------------------------------ #
#                    EXTRACT SPR_BENCH RESULTS                       #
# ------------------------------------------------------------------ #
ed = None
try:
    ed = experiment_data["NoRuleSparsity"]["SPR_BENCH"]
except Exception as e:
    print(f"Error extracting dataset: {e}")

if ed:
    train_loss = np.asarray(ed["losses"]["train"])
    val_loss = np.asarray(ed["losses"]["val"])
    train_acc = np.asarray(ed["metrics"]["train_acc"])
    val_acc = np.asarray(ed["metrics"]["val_acc"])
    rule_fid = np.asarray(ed["metrics"]["Rule_Fidelity"])
    preds = np.asarray(ed["predictions"])
    gts = np.asarray(ed["ground_truth"])
    epochs = np.arange(1, len(train_loss) + 1)

    # -------------------- Plot 1: Loss Curves ---------------------- #
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH – Loss Curves (NoRuleSparsity)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_Loss_Curves_NoRuleSparsity.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # -------------------- Plot 2: Accuracy Curves ------------------ #
    try:
        plt.figure()
        plt.plot(epochs, train_acc, label="Train")
        plt.plot(epochs, val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH – Accuracy Curves (NoRuleSparsity)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_Accuracy_Curves_NoRuleSparsity.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        plt.close()

    # -------------------- Plot 3: Rule Fidelity -------------------- #
    try:
        plt.figure()
        plt.plot(epochs, rule_fid, color="purple")
        plt.xlabel("Epoch")
        plt.ylabel("Rule Fidelity")
        plt.title("SPR_BENCH – Rule Fidelity over Epochs (NoRuleSparsity)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_RuleFidelity_NoRuleSparsity.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating rule-fidelity plot: {e}")
        plt.close()

    # -------------------- Plot 4: Confusion Matrix ----------------- #
    try:
        num_classes = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for gt, pr in zip(gts, preds):
            cm[gt, pr] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("SPR_BENCH – Confusion Matrix (NoRuleSparsity)")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "SPR_BENCH_ConfusionMatrix_NoRuleSparsity.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # -------------------- Print Key Metrics ------------------------ #
    try:
        test_acc = (preds == gts).mean()
        print(
            f"Final Val Acc: {val_acc[-1]:.3f} | "
            f"Test Acc: {test_acc:.3f} | "
            f"Final Rule Fidelity: {rule_fid[-1]:.3f}"
        )
    except Exception as e:
        print(f"Error computing metrics: {e}")
