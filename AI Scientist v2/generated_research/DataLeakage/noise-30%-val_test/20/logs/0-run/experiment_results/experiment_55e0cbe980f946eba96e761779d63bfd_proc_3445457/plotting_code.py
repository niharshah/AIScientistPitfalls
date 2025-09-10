import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------- iterate datasets ----------
for dname, logs in experiment_data.items():
    # basic sanity
    train_loss = np.array(logs["losses"].get("train", []), dtype=float)
    val_loss = np.array(logs["losses"].get("val", []), dtype=float)
    val_metrics = logs["metrics"].get("val", [])
    macro_f1 = (
        np.array([m["macro_f1"] for m in val_metrics], dtype=float)
        if val_metrics
        else np.array([])
    )
    cwa = (
        np.array([m["cwa"] for m in val_metrics], dtype=float)
        if val_metrics
        else np.array([])
    )
    preds = np.array(logs.get("predictions", []))
    gts = np.array(logs.get("ground_truth", []))
    wts = np.array(logs.get("weights", []))

    epochs = np.arange(1, len(train_loss) + 1)

    # 1) loss curves
    try:
        plt.figure()
        plt.plot(epochs, train_loss, label="train", color="tab:blue")
        plt.plot(epochs, val_loss, label="val", color="tab:orange", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"{dname}: Training vs Validation Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dname}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dname}: {e}")
        plt.close()

    # 2) macro-F1
    try:
        if macro_f1.size:
            plt.figure()
            plt.plot(epochs, macro_f1, color="tab:green")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(f"{dname}: Validation Macro-F1")
            plt.savefig(os.path.join(working_dir, f"{dname}_macro_f1.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating macro-F1 plot for {dname}: {e}")
        plt.close()

    # 3) CWA
    try:
        if cwa.size:
            plt.figure()
            plt.plot(epochs, cwa, color="tab:red")
            plt.xlabel("Epoch")
            plt.ylabel("Complexity-Weighted Acc.")
            plt.title(f"{dname}: Validation CWA")
            plt.savefig(os.path.join(working_dir, f"{dname}_cwa.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating CWA plot for {dname}: {e}")
        plt.close()

    # 4) confusion matrix
    try:
        if preds.size and gts.size:
            num_cls = int(max(preds.max(), gts.max()) + 1)
            cm = np.zeros((num_cls, num_cls), dtype=int)
            for p, t in zip(preds, gts):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i in range(num_cls):
                for j in range(num_cls):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{dname}: Confusion Matrix (final epoch)")
            plt.colorbar()
            plt.savefig(os.path.join(working_dir, f"{dname}_confusion_matrix.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {dname}: {e}")
        plt.close()

    # 5) weight histogram
    try:
        if wts.size:
            plt.figure()
            plt.hist(wts, bins=min(30, len(np.unique(wts))), color="tab:purple")
            plt.xlabel("Example Weight")
            plt.ylabel("Count")
            plt.title(f"{dname}: Distribution of Weights")
            plt.savefig(os.path.join(working_dir, f"{dname}_weight_hist.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating weight histogram for {dname}: {e}")
        plt.close()

    # ---- quick metric summary ----
    if macro_f1.size:
        print(
            f"{dname}: best Macro-F1={macro_f1.max():.3f} | final Macro-F1={macro_f1[-1]:.3f}"
        )
    if cwa.size:
        print(f"{dname}: best CWA={cwa.max():.3f} | final CWA={cwa[-1]:.3f}")
