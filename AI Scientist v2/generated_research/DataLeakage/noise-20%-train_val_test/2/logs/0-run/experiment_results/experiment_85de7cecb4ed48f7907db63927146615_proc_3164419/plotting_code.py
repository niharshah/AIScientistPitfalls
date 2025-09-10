import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# ------------------------- paths -------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------ load experiment data -----------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

keys = list(experiment_data.keys())
colors = plt.cm.tab10.colors

# ----------------- 1) Macro-F1 curves --------------------
try:
    plt.figure()
    for idx, k in enumerate(keys):
        epochs = experiment_data[k].get("epochs", [])
        tr_f1 = experiment_data[k]["metrics"].get("train_macro_f1", [])
        val_f1 = experiment_data[k]["metrics"].get("val_macro_f1", [])
        c = colors[idx % len(colors)]
        plt.plot(epochs, tr_f1, "--", color=c, label=f"{k}-train")
        plt.plot(epochs, val_f1, "-", color=c, label=f"{k}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curves (Train dashed, Validation solid)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_macro_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Macro-F1 plot: {e}")
    plt.close()

# ------------------- 2) Loss curves ----------------------
try:
    plt.figure()
    for idx, k in enumerate(keys):
        epochs = experiment_data[k].get("epochs", [])
        tr_loss = experiment_data[k]["losses"].get("train", [])
        val_loss = experiment_data[k]["losses"].get("val", [])
        c = colors[idx % len(colors)]
        plt.plot(epochs, tr_loss, "--", color=c, label=f"{k}-train")
        plt.plot(epochs, val_loss, "-", color=c, label=f"{k}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves (Train dashed, Validation solid)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Loss plot: {e}")
    plt.close()

# ---------------- 3) Test Macro-F1 bar chart -------------
try:
    test_scores = {k: experiment_data[k].get("test_macro_f1", np.nan) for k in keys}
    plt.figure()
    plt.bar(
        range(len(test_scores)),
        list(test_scores.values()),
        tick_label=list(test_scores.keys()),
    )
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Test Macro-F1 per Experiment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_test_macro_f1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating Test Macro-F1 bar: {e}")
    plt.close()

# -------- 4) Confusion matrix for test predictions -------
try:
    for idx, k in enumerate(keys[:5]):  # plot at most 5
        preds = experiment_data[k].get("predictions")
        gts = experiment_data[k].get("ground_truth")
        if preds is None or gts is None or len(preds) == 0:
            continue
        cm = confusion_matrix(gts, preds)
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{k} Confusion Matrix (Test Set)")
        plt.tight_layout()
        fname = os.path.join(working_dir, f"spr_bench_{k}_confusion_matrix.png")
        plt.savefig(fname)
        plt.close()
except Exception as e:
    print(f"Error creating Confusion Matrix: {e}")
    plt.close()

# ---------------- print numeric summary ------------------
print(
    "Test Macro-F1 scores:",
    {k: experiment_data[k].get("test_macro_f1", np.nan) for k in keys},
)
