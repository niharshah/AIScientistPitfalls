import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# quick sanity and key extraction
exp_key = next(iter(experiment_data.keys())) if experiment_data else None
bench_key = next(iter(experiment_data[exp_key].keys())) if exp_key else None

if exp_key and bench_key:
    data = experiment_data[exp_key][bench_key]
    epochs = np.array(data["epochs"])
    tr_loss = np.array(data["losses"]["train"])
    val_loss = np.array(data["losses"]["val"])
    tr_f1 = np.array(data["metrics"]["train_f1"])
    val_f1 = np.array(data["metrics"]["val_f1"])
    test_f1 = data["metrics"]["test_f1"]
    sga = data["metrics"]["SGA"]
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])

    # 1) Loss curves
    try:
        plt.figure()
        plt.plot(epochs, tr_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{bench_key}: Train vs Val Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{bench_key}_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # 2) F1 curves
    try:
        plt.figure()
        plt.plot(epochs, tr_f1, label="Train")
        plt.plot(epochs, val_f1, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.title(f"{bench_key}: Train vs Val Macro-F1")
        plt.legend()
        fname = os.path.join(working_dir, f"{bench_key}_f1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # 3) Bar chart of final metrics
    try:
        plt.figure()
        bars = ["Train_F1_last", "Val_F1_last", "Val_F1_best", "Test_F1", "SGA"]
        vals = [tr_f1[-1], val_f1[-1], val_f1.max(), test_f1, sga]
        plt.bar(bars, vals)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"{bench_key}: Summary Metrics")
        fname = os.path.join(working_dir, f"{bench_key}_metric_summary.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric bar chart: {e}")
        plt.close()

    # 4) Confusion matrix (truncate to first 15 classes to stay readable)
    try:
        num_classes = int(max(gts.max(), preds.max()) + 1)
        if num_classes <= 15:  # only plot if small enough
            cm = np.zeros((num_classes, num_classes), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues", interpolation="nearest")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{bench_key}: Confusion Matrix")
            plt.xticks(np.arange(num_classes))
            plt.yticks(np.arange(num_classes))
            fname = os.path.join(working_dir, f"{bench_key}_confusion_matrix.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ----- print key metrics -----
    print(f"Test macro-F1: {test_f1:.4f}, SGA: {sga:.4f}")
else:
    print("No experiment data found to plot.")
