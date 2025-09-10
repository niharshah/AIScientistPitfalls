import matplotlib.pyplot as plt
import numpy as np
import os

# working directory setup
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ----- load experiment data -----
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

exp_key = "MeanPool_NoRNN"
ds_key = "SPR_BENCH"

if exp_key in experiment_data and ds_key in experiment_data[exp_key]:
    data_block = experiment_data[exp_key][ds_key]
    losses_tr = data_block["losses"]["train"]
    losses_va = data_block["losses"]["val"]
    metrics_va = data_block["metrics"]["val"]  # list of dicts per epoch

    # -------------------- plot 1: loss curves --------------------
    try:
        epochs = range(1, len(losses_tr) + 1)
        plt.figure()
        plt.plot(epochs, losses_tr, label="Train Loss")
        plt.plot(epochs, losses_va, label="Val Loss")
        plt.title("SPR_BENCH Loss Curves (MeanPool_NoRNN)")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_MeanPool_NoRNN_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot: {e}")
        plt.close()

    # helper to extract metric list
    def extract_metric(metric_name):
        return [m[metric_name] for m in metrics_va]

    metric_names = ["acc", "CWA", "SWA", "CompWA"]
    for m_name in metric_names:
        try:
            plt.figure()
            plt.plot(epochs, extract_metric(m_name), marker="o")
            plt.title(f"SPR_BENCH {m_name.upper()} over Epochs (MeanPool_NoRNN)")
            plt.xlabel("Epoch")
            plt.ylabel(m_name.upper())
            plt.tight_layout()
            fname = os.path.join(working_dir, f"SPR_BENCH_MeanPool_NoRNN_{m_name}.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating {m_name} plot: {e}")
            plt.close()

    # ----- print final epoch metrics -----
    if metrics_va:
        final_metrics = metrics_va[-1]
        print("Final Validation Metrics:")
        for k, v in final_metrics.items():
            if k != "epoch":
                print(f"  {k}: {v:.4f}")
else:
    print("Requested experiment keys not found in experiment_data.")
