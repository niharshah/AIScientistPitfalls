import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def count_color_variety(sequence: str) -> int:
    return len(set(tok[1] for tok in sequence.strip().split() if len(tok) > 1))


def count_shape_variety(sequence: str) -> int:
    return len(set(tok[0] for tok in sequence.strip().split() if tok))


def pcwa(seqs, y_true, y_pred):
    w = [count_color_variety(s) * count_shape_variety(s) for s in seqs]
    return sum(wi if yt == yp else 0 for wi, yt, yp in zip(w, y_true, y_pred)) / max(
        sum(w), 1
    )


# ---------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to safely fetch nested dict
def safe_get(d, *keys, default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default


for exp_name, datasets in experiment_data.items():
    for ds_name, content in datasets.items():
        # --------------- Plot losses ---------------------------------
        try:
            losses_tr = safe_get(content, "losses", "train", default=[])
            losses_val = safe_get(content, "losses", "val", default=[])
            if losses_tr and losses_val:
                ep_tr, v_tr = zip(*losses_tr)
                ep_val, v_val = zip(*losses_val)
                plt.figure()
                plt.plot(ep_tr, v_tr, label="Train")
                plt.plot(ep_val, v_val, label="Validation")
                plt.title(f"{ds_name} Loss Curve (positional_feature_removal)")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                fname = f"{ds_name}_loss_curve_positional_feature_removal.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating loss curve: {e}")
            plt.close()

        # --------------- Plot PCWA -----------------------------------
        try:
            pc_tr = safe_get(content, "metrics", "train", default=[])
            pc_val = safe_get(content, "metrics", "val", default=[])
            if pc_tr and pc_val:
                ep_tr, v_tr = zip(*pc_tr)
                ep_val, v_val = zip(*pc_val)
                plt.figure()
                plt.plot(ep_tr, v_tr, label="Train")
                plt.plot(ep_val, v_val, label="Validation")
                plt.title(f"{ds_name} PCWA Curve (positional_feature_removal)")
                plt.xlabel("Epoch")
                plt.ylabel("PCWA")
                plt.legend()
                fname = f"{ds_name}_pcwa_curve_positional_feature_removal.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating PCWA curve: {e}")
            plt.close()

        # --------------- Confusion matrix ----------------------------
        try:
            y_true = content.get("ground_truth", [])
            y_pred = content.get("predictions", [])
            if y_true and y_pred:
                cm = np.zeros((2, 2), dtype=int)
                for yt, yp in zip(y_true, y_pred):
                    cm[yt, yp] += 1
                plt.figure()
                plt.imshow(cm, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
                plt.colorbar()
                plt.title(f"{ds_name} Confusion Matrix (Test set)")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                fname = f"{ds_name}_confusion_matrix_positional_feature_removal.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            plt.close()

        # --------------- Bar chart of test metrics -------------------
        try:
            if y_true and y_pred:
                seqs = content.get(
                    "sequences", []
                )  # not saved, so reuse ground truth len
                if not seqs:
                    seqs = [""] * len(y_true)
                acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)
                pc = pcwa(seqs, y_true, y_pred)
                # compute CWA & SWA
                cwa_num = sum(
                    count_color_variety(s) if yt == yp else 0
                    for s, yt, yp in zip(seqs, y_true, y_pred)
                )
                cwa_den = sum(count_color_variety(s) for s in seqs)
                swa_num = sum(
                    count_shape_variety(s) if yt == yp else 0
                    for s, yt, yp in zip(seqs, y_true, y_pred)
                )
                swa_den = sum(count_shape_variety(s) for s in seqs)
                cwa = cwa_num / max(cwa_den, 1)
                swa = swa_num / max(swa_den, 1)
                metrics = {"ACC": acc, "PCWA": pc, "CWA": cwa, "SWA": swa}

                plt.figure()
                plt.bar(metrics.keys(), metrics.values())
                plt.ylim(0, 1)
                plt.title(f"{ds_name} Test Metrics (positional_feature_removal)")
                fname = f"{ds_name}_test_metrics_bar_positional_feature_removal.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
                print(f"{ds_name} test metrics:", metrics)
        except Exception as e:
            print(f"Error creating metrics bar chart: {e}")
            plt.close()
