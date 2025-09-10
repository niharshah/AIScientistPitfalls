import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# ------------------------------------------------------------------
def count_color_variety(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def count_shape_variety(seq):
    return len({tok[0] for tok in seq.split()})


def cwa(seqs, y_t, y_p):
    w = [count_color_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def swa(seqs, y_t, y_p):
    w = [count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


def pcwa(seqs, y_t, y_p):
    w = [count_color_variety(s) + count_shape_variety(s) for s in seqs]
    c = [wt if t == p else 0 for wt, t, p in zip(w, y_t, y_p)]
    return sum(c) / sum(w) if sum(w) else 0.0


# ------------------------------------------------------------------
for model_name, data in experiment_data.items():
    # -------- loss curve --------------------------------------------------
    try:
        tr = np.array(data["losses"]["train"])
        vl = np.array(data["losses"]["val"])
        if tr.size and vl.size:
            plt.figure()
            plt.plot(tr[:, 0], tr[:, 1], label="Train")
            plt.plot(vl[:, 0], vl[:, 1], label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"SPR_BENCH {model_name}: Train vs Val Loss")
            plt.legend()
            fname = f"{model_name}_loss_curve_SPR_BENCH.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {model_name}: {e}")
        plt.close()

    # -------- metric curves ----------------------------------------------
    try:
        vmetrics = data["metrics"]["val"]
        if vmetrics:
            epochs = [e for e, _ in vmetrics]
            cwa_vals = [m["CWA"] for _, m in vmetrics]
            swa_vals = [m["SWA"] for _, m in vmetrics]
            pcwa_vals = [m["PCWA"] for _, m in vmetrics]
            plt.figure()
            plt.plot(epochs, cwa_vals, label="CWA")
            plt.plot(epochs, swa_vals, label="SWA")
            plt.plot(epochs, pcwa_vals, label="PCWA")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title(f"SPR_BENCH {model_name}: Validation Metrics")
            plt.legend()
            fname = f"{model_name}_val_metrics_SPR_BENCH.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error creating metric curve for {model_name}: {e}")
        plt.close()

    # -------- confusion heat-map ------------------------------------------
    try:
        gt = np.array(data.get("ground_truth", []))
        pr = np.array(data.get("predictions", []))
        if gt.size and pr.size:
            labels = sorted(set(gt) | set(pr))
            idx = {l: i for i, l in enumerate(labels)}
            mat = np.zeros((len(labels), len(labels)), dtype=int)
            for g, p in zip(gt, pr):
                mat[idx[g], idx[p]] += 1
            plt.figure()
            plt.imshow(mat, cmap="Blues")
            plt.colorbar()
            plt.xticks(range(len(labels)), labels, rotation=90)
            plt.yticks(range(len(labels)), labels)
            plt.title(
                f"SPR_BENCH {model_name}: Confusion Matrix\nLeft: Ground Truth, Right: Predicted"
            )
            fname = f"{model_name}_confusion_SPR_BENCH.png"
            plt.savefig(os.path.join(working_dir, fname), bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {model_name}: {e}")
        plt.close()

    # -------- print final test metrics ------------------------------------
    try:
        if gt.size and pr.size:
            acc = (gt == pr).mean()
            tcwa = cwa(gt, gt, pr)
            tswa = swa(gt, gt, pr)
            tpcwa = pcwa(gt, gt, pr)
            print(
                f"{model_name} TEST -> Accuracy {acc:.4f} | CWA {tcwa:.4f} | SWA {tswa:.4f} | PCWA {tpcwa:.4f}"
            )
    except Exception as e:
        print(f"Error computing final metrics for {model_name}: {e}")
