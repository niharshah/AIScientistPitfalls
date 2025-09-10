import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------------- SETUP
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "SPR_BENCH"
runs = experiment_data.get("d_model", {}).get(ds_name, {})

# ------------------------------------------------------------------------- FIG 1: LOSS CURVES
try:
    plt.figure()
    for d_model, log in runs.items():
        epochs = log["epochs"]
        plt.plot(
            epochs, log["losses"]["train"], label=f"{d_model}-train", linestyle="-"
        )
        plt.plot(epochs, log["losses"]["val"], label=f"{d_model}-val", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPR_BENCH: Training vs Validation Loss")
    plt.legend(fontsize="small")
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# ------------------------------------------------------------------------- FIG 2: F1 CURVES
try:
    plt.figure()
    for d_model, log in runs.items():
        epochs = log["epochs"]
        plt.plot(
            epochs, log["metrics"]["train_f1"], label=f"{d_model}-train", linestyle="-"
        )
        plt.plot(
            epochs, log["metrics"]["val_f1"], label=f"{d_model}-val", linestyle="--"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH: Training vs Validation Macro-F1")
    plt.legend(fontsize="small")
    fname = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# ------------------------------------------------------------------------- FIG 3: BEST VAL F1 PER MODEL SIZE
try:
    best_vals = {d: max(log["metrics"]["val_f1"]) for d, log in runs.items()}
    plt.figure()
    plt.bar(best_vals.keys(), best_vals.values(), color="skyblue")
    plt.xlabel("d_model")
    plt.ylabel("Best Validation Macro-F1")
    plt.title("SPR_BENCH: Best Val F1 by Model Size")
    fname = os.path.join(working_dir, "SPR_BENCH_best_val_f1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------------- FIG 4: CONFUSION MATRIX FOR BEST MODEL
try:
    # identify best model
    best_d = max(best_vals.items(), key=lambda kv: kv[1])[0]
    preds = np.array(runs[best_d]["predictions"])
    gts = np.array(runs[best_d]["ground_truth"])
    n_cls = int(max(preds.max(), gts.max())) + 1
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for p, g in zip(preds, gts):
        cm[g, p] += 1

    plt.figure()
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"SPR_BENCH Confusion Matrix (d_model={best_d})")
    fname = os.path.join(working_dir, f"SPR_BENCH_confusion_matrix_d{best_d}.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
