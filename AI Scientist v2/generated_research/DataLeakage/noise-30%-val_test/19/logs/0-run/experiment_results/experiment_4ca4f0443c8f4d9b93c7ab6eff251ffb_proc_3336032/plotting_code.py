import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths & data loading ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

# ---------- plotting ----------
max_exp_plots = 4  # keep total plots ≤5 (4 curves + 1 bar)
plotted = 0

if experiment_data:
    # per-experiment curve plots
    for exp_name, logs in experiment_data.items():
        if plotted >= max_exp_plots:
            break
        try:
            epochs = logs.get("epochs", [])
            tr_loss = logs.get("losses", {}).get("train", [])
            val_loss = logs.get("losses", {}).get("val", [])
            tr_mcc = logs.get("metrics", {}).get("train_MCC", [])
            val_mcc = logs.get("metrics", {}).get("val_MCC", [])

            plt.figure(figsize=(10, 4))
            # Left: loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("BCE Loss")
            plt.title("Loss")
            plt.legend()

            # Right: MCC
            plt.subplot(1, 2, 2)
            plt.plot(epochs, tr_mcc, label="Train")
            plt.plot(epochs, val_mcc, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("MCC")
            plt.title("Matthews Corr.")
            plt.legend()

            plt.suptitle(f"Left: Loss, Right: MCC — synthetic SPR_BENCH ({exp_name})")
            fname = f"spr_bench_{exp_name}_curves.png"
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
            plotted += 1
        except Exception as e:
            print(f"Error creating curve plot for {exp_name}: {e}")
            plt.close()

    # aggregated test-MCC bar chart
    try:
        names, test_scores = [], []
        for exp_name, logs in experiment_data.items():
            score = logs.get("test_MCC")
            if score is not None:
                names.append(exp_name)
                test_scores.append(score)
        if test_scores:
            plt.figure(figsize=(6, 4))
            plt.bar(names, test_scores, color="lightgreen")
            plt.xlabel("Experiment")
            plt.ylabel("Test MCC")
            plt.title("Test MCC by Experiment — synthetic SPR_BENCH")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "spr_bench_test_mcc_bar.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated bar chart: {e}")
        plt.close()
