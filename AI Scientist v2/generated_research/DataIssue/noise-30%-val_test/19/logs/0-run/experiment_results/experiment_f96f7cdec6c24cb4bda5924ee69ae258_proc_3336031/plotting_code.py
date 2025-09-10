import matplotlib.pyplot as plt
import numpy as np
import os

# ------------ paths & data loading -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    # ---------- per-config curves (≤5 figs) ----------
    for tag, logs in experiment_data.items():
        try:
            epochs = logs.get("epochs", [])
            tr_loss = logs.get("losses", {}).get("train", [])
            val_loss = logs.get("losses", {}).get("val", [])
            tr_mcc = logs.get("metrics", {}).get("train_MCC", [])
            val_mcc = logs.get("metrics", {}).get("val_MCC", [])

            plt.figure(figsize=(10, 4))
            # Left panel: Loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("BCE Loss")
            plt.title("Loss")
            plt.legend()

            # Right panel: MCC
            plt.subplot(1, 2, 2)
            plt.plot(epochs, tr_mcc, label="Train")
            plt.plot(epochs, val_mcc, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("MCC")
            plt.title("MCC")
            plt.legend()

            plt.suptitle(
                f"Left: Train/Val Loss, Right: Train/Val MCC — SPR_BENCH ({tag})"
            )
            fname = f"spr_bench_loss_mcc_{tag}.png"
            plt.tight_layout(rect=[0, 0.04, 1, 0.94])
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating curve plot for {tag}: {e}")
            plt.close()

    # ---------- aggregated bar chart of test MCC ----------
    try:
        cfgs, test_mccs = [], []
        for tag, logs in experiment_data.items():
            score = logs.get("test_MCC")
            if score is not None:
                cfgs.append(tag)
                test_mccs.append(score)

        if test_mccs:
            plt.figure(figsize=(6, 4))
            plt.bar(cfgs, test_mccs, color="orange")
            plt.xlabel("Config")
            plt.ylabel("Test MCC")
            plt.title("Test MCC by Config — SPR_BENCH")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "spr_bench_test_MCC_bar.png"))
            plt.close()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()
