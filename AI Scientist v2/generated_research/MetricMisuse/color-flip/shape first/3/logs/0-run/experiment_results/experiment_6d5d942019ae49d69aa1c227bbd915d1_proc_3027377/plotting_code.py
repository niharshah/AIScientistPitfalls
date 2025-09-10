import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
# directory housekeeping
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None


# ---------------------------------------------------------------------
# helper metrics (re-implemented here)
def count_shape_variety(seq: str) -> int:
    return len(set(tok[0] for tok in seq.split()))


def count_color_variety(seq: str) -> int:
    return len(set(tok[1] if len(tok) > 1 else "#" for tok in seq.split()))


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) + 1e-9
    )


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    return sum(wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)) / (
        sum(w) + 1e-9
    )


# ---------------------------------------------------------------------
if experiment_data is not None:
    for ds_name, rec in experiment_data.items():
        # ---------- recover arrays ----------
        train_losses = rec["losses"]["train"]
        val_losses = rec["losses"]["val"]
        val_acc = rec["metrics"].get("val_acc", [])
        val_swa = rec["metrics"].get("val_swa", [])
        val_cwa = rec["metrics"].get("val_cwa", [])
        predictions = rec.get("predictions", [])
        ground_truth = rec.get("ground_truth", [])
        aca_val = rec["aca"].get("val", [])
        aca_test = rec["aca"].get("test", np.nan)

        epochs = np.arange(1, len(train_losses) + 1)

        # ---------- Plot 1: loss curves ----------
        try:
            plt.figure()
            plt.plot(epochs, train_losses, "r--", label="Train")
            plt.plot(epochs, val_losses, "b-", label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-entropy loss")
            plt.title(f"{ds_name}: Training & Validation Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_loss_curves.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {ds_name}: {e}")
            plt.close()

        # ---------- Plot 2: validation metric curves ----------
        try:
            plt.figure()
            if val_acc:
                plt.plot(epochs, val_acc, "g-", label="Acc")
            if val_swa:
                plt.plot(epochs, val_swa, "m-", label="SWA")
            if val_cwa:
                plt.plot(epochs, val_cwa, "c-", label="CWA")
            if aca_val:
                plt.plot(epochs, aca_val, "k--", label="ACA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"{ds_name}: Validation Metrics vs Epoch")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds_name}_val_metrics.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error creating metric plot for {ds_name}: {e}")
            plt.close()

        # ---------- compute & print test metrics ----------
        if predictions and ground_truth:
            # we lack raw sequences for SWA/CWA on test, so reuse preds only for ACC
            acc = np.mean(np.array(predictions) == np.array(ground_truth))
            # try to extract raw seqs stored during evaluation if present
            raw_seqs = rec.get("raw_test_seqs", [""] * len(predictions))
            swa = shape_weighted_accuracy(raw_seqs, ground_truth, predictions)
            cwa = color_weighted_accuracy(raw_seqs, ground_truth, predictions)
            print(
                f"{ds_name} Test metrics — ACC: {acc:.4f} | SWA: {swa:.4f} | "
                f"CWA: {cwa:.4f} | ACA: {aca_test:.4f}"
            )

            # ---------- Plot 3: bar summary ----------
            try:
                plt.figure()
                metrics = ["ACC", "SWA", "CWA", "ACA"]
                vals = [acc, swa, cwa, aca_test]
                plt.bar(metrics, vals, color=["g", "m", "c", "k"], alpha=0.7)
                plt.ylim(0, 1)
                plt.ylabel("Score")
                plt.title(f"{ds_name}: Test Metric Summary")
                fname = os.path.join(working_dir, f"{ds_name}_test_summary.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
                plt.close()
            except Exception as e:
                print(f"Error creating summary bar plot for {ds_name}: {e}")
                plt.close()

            # ---------- Plot 4: confusion matrix ----------
            try:
                classes = sorted(set(ground_truth))
                cm = np.zeros((len(classes), len(classes)), dtype=int)
                for t, p in zip(ground_truth, predictions):
                    cm[t, p] += 1
                plt.figure()
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title(f"{ds_name}: Confusion Matrix")
                plt.xticks(classes)
                plt.yticks(classes)
                fname = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
                plt.savefig(fname)
                print(f"Saved {fname}")
                plt.close()
            except Exception as e:
                print(f"Error creating confusion matrix for {ds_name}: {e}")
                plt.close()
