import matplotlib.pyplot as plt
import numpy as np
import os

# --- IO ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# --- helpers ---
def safe_fig(plot_fn, fname):
    try:
        plot_fn()
        plt.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error creating {fname}: {e}")
    finally:
        plt.close()


# --- iterate over stored experiments / datasets ---
for exp_name, exp_rec in experiment_data.items():
    for ds_name, ds_rec in exp_rec.items():
        tr_loss = ds_rec["losses"]["train"]
        va_loss = ds_rec["losses"]["val"]
        metrics = ds_rec["metrics"]["val"]  # list of dicts
        epochs = range(1, len(tr_loss) + 1)

        # stack metrics
        acc = [m["acc"] for m in metrics]
        cwa = [m["CWA"] for m in metrics]
        swa = [m["SWA"] for m in metrics]
        comp = [m["CompWA"] for m in metrics]

        # 1. loss curve
        safe_fig(
            lambda: (
                plt.figure(),
                plt.plot(epochs, tr_loss, label="train"),
                plt.plot(epochs, va_loss, label="val"),
                plt.title(f"{ds_name} Loss Curve ({exp_name})"),
                plt.xlabel("Epoch"),
                plt.ylabel("Loss"),
                plt.legend(),
            ),
            f"{ds_name}_{exp_name}_loss.png",
        )

        # 2. accuracy
        safe_fig(
            lambda: (
                plt.figure(),
                plt.plot(epochs, acc, marker="o"),
                plt.title(f"{ds_name} Validation Accuracy ({exp_name})"),
                plt.xlabel("Epoch"),
                plt.ylabel("Accuracy"),
            ),
            f"{ds_name}_{exp_name}_val_acc.png",
        )

        # 3. CWA
        safe_fig(
            lambda: (
                plt.figure(),
                plt.plot(epochs, cwa, marker="o"),
                plt.title(f"{ds_name} Color-Weighted Acc ({exp_name})"),
                plt.xlabel("Epoch"),
                plt.ylabel("CWA"),
            ),
            f"{ds_name}_{exp_name}_CWA.png",
        )

        # 4. SWA
        safe_fig(
            lambda: (
                plt.figure(),
                plt.plot(epochs, swa, marker="o"),
                plt.title(f"{ds_name} Shape-Weighted Acc ({exp_name})"),
                plt.xlabel("Epoch"),
                plt.ylabel("SWA"),
            ),
            f"{ds_name}_{exp_name}_SWA.png",
        )

        # 5. Confusion matrix (optional, plotted once)
        if ds_rec.get("predictions") and ds_rec.get("ground_truth"):
            y_true = np.array(ds_rec["ground_truth"])
            y_pred = np.array(ds_rec["predictions"])
            k = int(max(y_true.max(), y_pred.max()) + 1)
            cm = np.zeros((k, k), int)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            safe_fig(
                lambda: (
                    plt.figure(),
                    plt.imshow(cm, cmap="Blues"),
                    plt.colorbar(),
                    plt.title(f"{ds_name} Confusion Matrix ({exp_name})"),
                    plt.xlabel("Predicted"),
                    plt.ylabel("True"),
                ),
                f"{ds_name}_{exp_name}_confmat.png",
            )

        # print last-epoch snapshot
        print(
            f"{exp_name}/{ds_name} – Epoch {len(epochs)}: "
            f"ACC={acc[-1]:.3f} CWA={cwa[-1]:.3f} SWA={swa[-1]:.3f} CompWA={comp[-1]:.3f}"
        )
