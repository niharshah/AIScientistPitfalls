import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


def count_color_variety(seq):
    return len({t[1] for t in seq.split() if len(t) > 1})


def count_shape_variety(seq):
    return len({t[0] for t in seq.split() if t})


def cwa(S, y, yh):
    w = [count_color_variety(s) for s in S]
    return sum(wt if a == b else 0 for wt, a, b in zip(w, y, yh)) / sum(w)


def swa(S, y, yh):
    w = [count_shape_variety(s) for s in S]
    return sum(wt if a == b else 0 for wt, a, b in zip(w, y, yh)) / sum(w)


def pcwa(S, y, yh):
    w = [count_color_variety(s) + count_shape_variety(s) for s in S]
    return sum(wt if a == b else 0 for wt, a, b in zip(w, y, yh)) / sum(w)


try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "shape_color_transformer" in experiment_data:
    data = experiment_data["shape_color_transformer"]
    # ---------- figure 1: loss curves ----------
    try:
        tr_epochs, tr_losses = zip(*data["losses"]["train"])
        val_epochs, val_losses = zip(*data["losses"]["val"])
        plt.figure()
        plt.plot(tr_epochs, tr_losses, label="Train")
        plt.plot(val_epochs, val_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss Curve\nLeft: Train, Right: Val")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, "loss_curve_SPR_BENCH_shape_color_transformer.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # prepare metric dicts by epoch
    metric_epochs = []
    cwas = []
    swas = []
    pcwas = []
    for ep, md in data["metrics"]["val"]:
        metric_epochs.append(ep)
        cwas.append(md["CWA"])
        swas.append(md["SWA"])
        pcwas.append(md["PCWA"])

    # ---------- figures 2-4: metric curves ----------
    for name, vals in [("CWA", cwas), ("SWA", swas), ("PCWA", pcwas)]:
        try:
            plt.figure()
            plt.plot(metric_epochs, vals, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel(name)
            plt.title(f"SPR_BENCH Validation {name} Across Epochs")
            fname = f"val_{name}_curve_SPR_BENCH_shape_color_transformer.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting {name}: {e}")
            plt.close()

    # ---------- figure 5: bar chart final val vs test ----------
    try:
        # final val metrics
        final_val = {"CWA": cwas[-1], "SWA": swas[-1], "PCWA": pcwas[-1]}
        # recompute test metrics
        seqs = experiment_data["shape_color_transformer"].get(
            "ground_truth", []
        )  # placeholder length check
        if seqs:
            y_true = experiment_data["shape_color_transformer"]["ground_truth"]
            y_pred = experiment_data["shape_color_transformer"]["predictions"]
            tcwa, tswa, tpcwa = (
                cwa(seqs, y_true, y_pred),
                swa(seqs, y_true, y_pred),
                pcwa(seqs, y_true, y_pred),
            )
            test_metrics = {"CWA": tcwa, "SWA": tswa, "PCWA": tpcwa}
        else:
            test_metrics = {"CWA": 0, "SWA": 0, "PCWA": 0}

        labels = list(final_val.keys())
        x = np.arange(len(labels))
        width = 0.35
        plt.figure()
        plt.bar(x - width / 2, [final_val[l] for l in labels], width, label="Val")
        plt.bar(x + width / 2, [test_metrics[l] for l in labels], width, label="Test")
        plt.xticks(x, labels)
        plt.ylabel("Score")
        plt.title("SPR_BENCH Final Metrics\nLeft: Validation, Right: Test")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, "metric_comparison_SPR_BENCH_shape_color_transformer.png"
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating metric comparison plot: {e}")
        plt.close()

    # ---------- print final metrics ----------
    print("Final Validation Metrics:", final_val)
    print("Test Metrics:", test_metrics)
