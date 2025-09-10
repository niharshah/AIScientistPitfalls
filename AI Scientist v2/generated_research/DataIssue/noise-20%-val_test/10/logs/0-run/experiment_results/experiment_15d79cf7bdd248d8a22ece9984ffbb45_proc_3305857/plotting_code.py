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


# ---------- helpers ----------
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


if experiment_data:
    best_val_f1_by_ds = {}
    for dname, ds in experiment_data.items():
        # ---------- 1) Train / Val macro-F1 ----------
        try:
            tr = np.array(ds["metrics"]["train"])
            val = np.array(ds["metrics"]["val"])
            epochs = tr[:, 0]
            plt.figure()
            plt.plot(epochs, tr[:, 2], label="train_F1")
            plt.plot(epochs, val[:, 2], label="val_F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{dname}: Train vs Val Macro-F1")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_f1_curve.png")
            plt.savefig(fname)
            plt.close()
            best_val_f1_by_ds[dname] = val[:, 2].max()
        except Exception as e:
            print(f"Error creating F1 curve for {dname}: {e}")
            plt.close()

        # ---------- 2) Train / Val loss ----------
        try:
            tr_loss = np.array(ds["losses"]["train"])
            val_loss = np.array(ds["losses"]["val"])
            epochs = np.arange(1, len(tr_loss) + 1)
            plt.figure()
            plt.plot(epochs, tr_loss, label="train_loss")
            plt.plot(epochs, val_loss, label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"{dname}: Loss Curves")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_loss_curve.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve for {dname}: {e}")
            plt.close()

        # ---------- 3) Confusion matrix ----------
        try:
            preds = np.array(ds["predictions"])
            gts = np.array(ds["ground_truth"])
            if preds.size and gts.size:
                labels = np.unique(np.concatenate([gts, preds]))
                cm = np.zeros((len(labels), len(labels)), int)
                for t, p in zip(gts, preds):
                    cm[t, p] += 1
                plt.figure(figsize=(6, 5))
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"{dname}: Confusion Matrix (Test)")
                plt.xticks(labels)
                plt.yticks(labels)
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
                fname = os.path.join(working_dir, f"{dname}_confusion_matrix.png")
                plt.savefig(fname)
                plt.close()
                print(f"{dname} Test Macro-F1:", macro_f1(gts, preds))
            else:
                raise ValueError("empty preds/gts")
        except Exception as e:
            print(f"Error creating confusion matrix for {dname}: {e}")
            plt.close()

        # ---------- 4) Rule histogram ----------
        try:
            rules = np.array(ds.get("rules", []))
            if rules.size:
                plt.figure()
                plt.hist(rules, bins=min(50, len(np.unique(rules))), color="gray")
                plt.xlabel("Rule ID")
                plt.ylabel("Frequency")
                plt.title(f"{dname}: Distribution of Selected Rules")
                fname = os.path.join(working_dir, f"{dname}_rule_histogram.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating rule histogram for {dname}: {e}")
            plt.close()

    # ---------- 5) Cross-dataset comparison (best val F1) ----------
    if len(best_val_f1_by_ds) > 1:
        try:
            plt.figure()
            names = list(best_val_f1_by_ds.keys())
            vals = [best_val_f1_by_ds[n] for n in names]
            plt.bar(names, vals, color="skyblue")
            plt.ylabel("Best Validation Macro-F1")
            plt.title("Dataset Comparison: Peak Validation F1")
            plt.xticks(rotation=45, ha="right")
            fname = os.path.join(working_dir, "datasets_best_val_f1_comparison.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error creating cross-dataset comparison: {e}")
            plt.close()
