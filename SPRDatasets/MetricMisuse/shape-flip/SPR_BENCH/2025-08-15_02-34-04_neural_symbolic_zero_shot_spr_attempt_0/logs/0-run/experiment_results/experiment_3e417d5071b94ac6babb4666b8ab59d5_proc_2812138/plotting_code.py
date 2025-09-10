import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----------- load experiment data -----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["num_epochs"]["spr_bench"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = None

if spr_data:
    runs_train_loss = spr_data["losses"]["train"]
    runs_val_loss = spr_data["losses"]["val"]
    runs_train_met = spr_data["metrics"]["train"]  # list[list[tuple]]
    runs_val_met = spr_data["metrics"]["val"]
    val_hwa_final = spr_data["val_hwa"]
    chosen_epochs = spr_data["chosen_epochs"]
    y_pred_test = np.array(spr_data["predictions"])
    y_true_test = np.array(spr_data["ground_truth"])

    # helper to extract HWA curve
    def hwa_curve(metrics_per_epoch):
        return [m[2] for m in metrics_per_epoch]

    # ------------- Plot 1: loss curves -------------
    try:
        fig, ax = plt.subplots()
        for i, (tr, va) in enumerate(zip(runs_train_loss, runs_val_loss)):
            ax.plot(tr, label=f"run{i}_train")
            ax.plot(va, linestyle="--", label=f"run{i}_val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("SPR_BENCH – Train/Val Loss per Run")
        ax.legend(fontsize=6)
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
    finally:
        plt.close()

    # ------------- Plot 2: HWA curves -------------
    try:
        fig, ax = plt.subplots()
        for i, (tr_m, va_m) in enumerate(zip(runs_train_met, runs_val_met)):
            ax.plot(hwa_curve(tr_m), label=f"run{i}_train")
            ax.plot(hwa_curve(va_m), linestyle="--", label=f"run{i}_val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("HWA")
        ax.set_title("SPR_BENCH – Train/Val HWA per Run")
        ax.legend(fontsize=6)
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_hwa_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating HWA curve plot: {e}")
    finally:
        plt.close()

    # ------------- Plot 3: Final val HWA bar chart -------------
    try:
        fig, ax = plt.subplots()
        bars = np.arange(len(val_hwa_final))
        ax.bar(bars, val_hwa_final, color="steelblue")
        ax.set_xticks(bars)
        ax.set_xticklabels([f"run{i}" for i in bars], rotation=45)
        ax.set_ylabel("Final Validation HWA")
        ax.set_title("SPR_BENCH – Final Validation HWA by Run")
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_val_hwa_bar.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating bar chart: {e}")
    finally:
        plt.close()

    # ------------- Plot 4: Confusion matrix -------------
    try:
        labels = sorted(list(set(y_true_test)))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_true_test, y_pred_test):
            cm[label_to_idx[t], label_to_idx[p]] += 1
        cm_norm = cm / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("SPR_BENCH – Confusion Matrix (Test)")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fname = os.path.join(working_dir, "spr_bench_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
    finally:
        plt.close()
