import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["ORD_EMB_CLUSTER_ABLATION"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

saved_plots = []

if exp:
    # ---------- pre-extract common arrays ----------
    tr_loss = np.asarray(exp["losses"]["train"], dtype=float)
    val_loss = np.asarray(exp["losses"]["val"], dtype=float)
    val_metrics = exp["metrics"]["val"]
    epochs = np.arange(1, len(tr_loss) + 1)

    # prepare val metric arrays if they exist
    def get_metric(m):
        return np.asarray([d.get(m, np.nan) for d in val_metrics], dtype=float)

    acc, cwa, swa, ccwa = map(get_metric, ["acc", "cwa", "swa", "ccwa"])

    # ---------- Plot 1: loss curves ----------
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        saved_plots.append(fname)
    except Exception as e:
        print(f"Error creating loss curve: {e}")
    finally:
        plt.close()

    # ---------- Plot 2: validation metrics over epochs ----------
    try:
        plt.figure()
        for arr, lab in zip([acc, cwa, swa, ccwa], ["ACC", "CWA", "SWA", "CCWA"]):
            if not np.all(np.isnan(arr)):
                plt.plot(epochs, arr, label=lab)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Validation Metrics across Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
        plt.savefig(fname)
        saved_plots.append(fname)
    except Exception as e:
        print(f"Error creating val metrics plot: {e}")
    finally:
        plt.close()

    # ---------- Plot 3: test metrics bar chart ----------
    try:
        test_m = exp["metrics"]["test"]
        labels = list(test_m.keys())
        values = [test_m[k] for k in labels]
        plt.figure()
        plt.bar(labels, values, color="skyblue")
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.title("SPR_BENCH: Test Metrics Summary")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        saved_plots.append(fname)
    except Exception as e:
        print(f"Error creating test metrics bar: {e}")
    finally:
        plt.close()

    # ---------- Plot 4: confusion matrix ----------
    try:
        gt = np.asarray(exp["ground_truth"]).ravel()
        pr = np.asarray(exp["predictions"]).ravel()
        if gt.size and pr.size and gt.shape == pr.shape:
            num_classes = int(max(gt.max(), pr.max())) + 1
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for g, p in zip(gt, pr):
                cm[g, p] += 1
            plt.figure()
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title("SPR_BENCH: Confusion Matrix\nLeft: GT, Right: Predicted")
            plt.xticks(range(num_classes))
            plt.yticks(range(num_classes))
            fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fname)
            saved_plots.append(fname)
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
    finally:
        plt.close()

print("Saved plots:", saved_plots)
