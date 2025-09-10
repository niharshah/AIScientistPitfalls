import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- directories ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    spr = experiment_data["SPR_BENCH"]

    # ---------- helper ----------
    def macro_f1(y_true, y_pred):
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1s = []
        for lb in labels:
            tp = np.sum((y_true == lb) & (y_pred == lb))
            fp = np.sum((y_true != lb) & (y_pred == lb))
            fn = np.sum((y_true == lb) & (y_pred != lb))
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        return np.mean(f1s)

    # ---------- metrics ----------
    preds = np.array(spr.get("preds_test", []))
    gts = np.array(spr.get("gts_test", []))
    if len(preds) and len(gts):
        print("Test Macro-F1:", macro_f1(gts, preds))

    # ---------- 1) Loss curves ----------
    try:
        tr_loss = spr["losses"]["train"]
        val_loss = spr["losses"]["val"]
        epochs = np.arange(1, len(tr_loss) + 1)
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves: {e}")
        plt.close()

    # ---------- 2) Macro-F1 curves ----------
    try:
        tr_f1 = spr["metrics"]["train_f1"]
        val_f1 = spr["metrics"]["val_f1"]
        epochs = np.arange(1, len(tr_f1) + 1)
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train Macro-F1")
        plt.plot(epochs, val_f1, label="Validation Macro-F1")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 Curves")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating f1 curves: {e}")
        plt.close()

    # ---------- 3) Confusion matrix ----------
    try:
        if len(preds) and len(gts):
            labels = np.unique(np.concatenate([gts, preds]))
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im)
            plt.title("SPR_BENCH Confusion Matrix (Test)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(labels)
            plt.yticks(labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- 4) Extracted rule characters ----------
    try:
        rules = spr.get("rules", {})
        if rules:
            classes = list(rules.keys())
            toks = [rules[c] for c in classes]
            plt.figure()
            plt.bar(classes, np.ones(len(classes)))
            plt.xticks(classes, toks)
            plt.ylabel("Presence (dummy value)")
            plt.title("SPR_BENCH: Extracted Rule Token per Class")
            fname = os.path.join(working_dir, "SPR_BENCH_rule_tokens.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating rule token plot: {e}")
        plt.close()
