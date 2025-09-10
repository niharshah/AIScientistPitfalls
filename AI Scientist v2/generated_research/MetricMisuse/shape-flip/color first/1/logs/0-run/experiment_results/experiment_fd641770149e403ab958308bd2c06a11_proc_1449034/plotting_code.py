import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split() if len(tok) > 1))


def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split() if tok))


def color_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_color_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / max(sum(w), 1)


def harmonic_poly_accuracy(cwa, swa):
    return 2 * cwa * swa / (cwa + swa + 1e-8)


# ------------- load experiment data -------------
try:
    exp_path = os.path.join(working_dir, "experiment_data.npy")
    experiment_data = np.load(exp_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    exp = experiment_data["no_homophily_edges"]["SPR"]
    epochs = exp["epochs"]
    tr_loss = exp["losses"]["train"]
    val_loss = exp["losses"]["val"]
    metrics = exp["metrics"]["val"]  # list of dicts

    # ------ Plot 1: loss curves ------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR Dataset – Loss Curve (no_homophily_edges)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_loss_curve_no_homophily_edges.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # ------ Plot 2: validation metrics ------
    try:
        cwa = [m["CWA"] for m in metrics]
        swa = [m["SWA"] for m in metrics]
        hpa = [m["HPA"] for m in metrics]
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, hpa, label="HPA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR Dataset – Validation Metrics (no_homophily_edges)")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_val_metrics_no_homophily_edges.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric curves: {e}")
        plt.close()

    # ------ Plot 3: confusion matrix on test set ------
    try:
        preds = np.array(exp["predictions"])
        gts = np.array(exp["ground_truth"])
        num_cls = int(max(gts.max(), preds.max()) + 1) if len(gts) else 2
        conf = np.zeros((num_cls, num_cls), dtype=int)
        for t, p in zip(gts, preds):
            conf[t, p] += 1
        plt.figure()
        plt.imshow(conf, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR Dataset – Confusion Matrix (no_homophily_edges)")
        for i in range(num_cls):
            for j in range(num_cls):
                plt.text(j, i, conf[i, j], ha="center", va="center", color="black")
        fname = os.path.join(working_dir, "SPR_confusion_matrix_no_homophily_edges.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ------ Print final test metrics ------
    try:
        seqs = (
            exp.get("seqs_test") if "seqs_test" in exp else []
        )  # fallback if not stored
        # original exp stored raw seqs only if saved; if not, we can't recompute
        if not seqs and "ground_truth" in exp:
            print("Raw sequences unavailable; only confusion matrix printed above.")
        else:
            cwa_final = color_weighted_accuracy(seqs, gts, preds)
            swa_final = shape_weighted_accuracy(seqs, gts, preds)
            hpa_final = harmonic_poly_accuracy(cwa_final, swa_final)
            print(
                f"Final TEST metrics -> CWA={cwa_final:.3f} | SWA={swa_final:.3f} | HPA={hpa_final:.3f}"
            )
    except Exception as e:
        print(f"Error computing final metrics: {e}")
