import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    weights = [count_shape_variety(s) for s in seqs]
    correct = [w if t == p else 0 for w, t, p in zip(weights, y_true, y_pred)]
    return sum(correct) / max(sum(weights), 1e-6)


# --------------------------------------------------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["REMOVE_COLOR"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

# ------------ 1) Training / Validation Loss curves -------------- #
try:
    ep_tr, loss_tr = (
        zip(*exp["losses"]["train"]) if exp["losses"]["train"] else ([], [])
    )
    ep_val, loss_val = zip(*exp["losses"]["val"]) if exp["losses"]["val"] else ([], [])
    plt.figure()
    plt.plot(ep_tr, loss_tr, label="Train")
    plt.plot(ep_val, loss_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH – Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# ------------ 2) Training / Validation SWA curves --------------- #
try:
    ep_tr, swa_tr = (
        zip(*exp["metrics"]["train"]) if exp["metrics"]["train"] else ([], [])
    )
    ep_val, swa_val = zip(*exp["metrics"]["val"]) if exp["metrics"]["val"] else ([], [])
    plt.figure()
    plt.plot(ep_tr, swa_tr, label="Train")
    plt.plot(ep_val, swa_val, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Shape-Weighted Accuracy")
    plt.title("SPR_BENCH – Training vs Validation SWA")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_SWA_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SWA curve plot: {e}")
    plt.close()

# ------------------- 3) Confusion Matrix ------------------------ #
try:
    preds = np.array(exp["predictions"])
    gts = np.array(exp["ground_truth"])
    if preds.size and gts.size:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gts, preds):
            cm[t, p] += 1
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH – Confusion Matrix (Test Set)")
        plt.colorbar()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png"))
        plt.close()
    else:
        print("Predictions / ground-truth arrays are empty, skipping confusion matrix.")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ------------------- Print test SWA ----------------------------- #
try:
    if preds.size and gts.size:
        # need raw sequences for SWA
        raw_seqs = exp.get("raw_sequences", [])  # may not exist
        if raw_seqs and len(raw_seqs) == len(preds):
            test_swa = shape_weighted_accuracy(raw_seqs, gts.tolist(), preds.tolist())
            print(f"Test Shape-Weighted Accuracy: {test_swa:.4f}")
except Exception as e:
    print(f"Error computing test SWA: {e}")
