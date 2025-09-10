import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

test_hwa = {}
if experiment_data is not None:
    sweep = experiment_data.get("tune_hidden_dim", {})
    # ---------------------------------------------------#
    # Individual loss curves (max 4 -> one per hidden dim)
    # ---------------------------------------------------#
    for run_key, data in list(sweep.items())[:4]:  # safety slice in case of extra runs
        try:
            tr_loss = data["losses"]["train"]
            val_loss = data["losses"]["val"]
            epochs = range(1, len(tr_loss) + 1)
            plt.figure()
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(f"SPR Loss Curves – {run_key} (Dataset: SPR)")
            plt.legend()
            fname = f"SPR_loss_curve_{run_key}.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting {run_key} losses: {e}")
            plt.close()
        # store test metric for later bar plot
        try:
            test_hwa[run_key] = data["metrics"]["test"]
        except KeyError:
            pass

    # ----------------------------------------------#
    # Summary bar chart of final test HWA per config
    # ----------------------------------------------#
    try:
        if test_hwa:
            plt.figure()
            keys = list(test_hwa.keys())
            vals = [test_hwa[k] for k in keys]
            plt.bar(keys, vals, color="steelblue")
            plt.ylabel("Test HWA")
            plt.ylim(0, 1)
            plt.title("SPR Test HWA vs Hidden Dim (Dataset: SPR)")
            for i, v in enumerate(vals):
                plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
            fname = "SPR_test_HWA_vs_hidden_dim.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
    except Exception as e:
        print(f"Error plotting summary bar chart: {e}")
        plt.close()

    # ------------------------------#
    # Console print of test metrics #
    # ------------------------------#
    for k, v in test_hwa.items():
        print(f"{k}: Test HWA = {v:.4f}")
