import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    run = experiment_data["bag_of_words_encoder"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    run = None

if run:
    epochs_tr = list(range(1, len(run["losses"]["train"]) + 1))
    epochs_val = list(range(1, len(run["losses"]["val"]) + 1))
    train_loss = run["losses"]["train"]
    val_loss = run["losses"]["val"]
    val_metrics = run["metrics"]["val"]

    # Plot 1: Train vs Val loss
    try:
        plt.figure()
        plt.plot(epochs_tr, train_loss, label="Train Loss")
        plt.plot(epochs_val, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH: Training vs Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # Plot 2: SWA / CWA / CCWA over epochs
    try:
        plt.figure()
        swa = [m["swa"] for m in val_metrics]
        cwa = [m["cwa"] for m in val_metrics]
        ccwa = [m["ccwa"] for m in val_metrics]
        plt.plot(epochs_val, swa, label="SWA")
        plt.plot(epochs_val, cwa, label="CWA")
        plt.plot(epochs_val, ccwa, label="CCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("SPR_BENCH: Weighted Accuracy Metrics over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_metrics_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metrics curve: {e}")
        plt.close()

    # Plot 3: Final metric comparison (best epoch = last stored in predictions)
    try:
        plt.figure()
        best_idx = len(run["predictions"]) > 0  # predictions stored only for best epoch
        if best_idx:
            best_metrics = val_metrics[
                [m["ccwa"] for m in val_metrics].index(
                    max([m["ccwa"] for m in val_metrics])
                )
            ]
        else:
            best_metrics = val_metrics[-1]
        labels = ["SWA", "CWA", "CCWA"]
        values = [best_metrics["swa"], best_metrics["cwa"], best_metrics["ccwa"]]
        plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Final Validation Metrics (Best Epoch)")
        fname = os.path.join(working_dir, "SPR_BENCH_final_metrics_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating final metrics bar: {e}")
        plt.close()
