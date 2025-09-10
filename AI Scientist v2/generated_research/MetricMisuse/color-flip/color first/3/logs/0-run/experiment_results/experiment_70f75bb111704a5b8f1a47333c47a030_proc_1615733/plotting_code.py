import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------- load data ------------------- #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper: safe close even on error
def save_plot(fig, fname):
    try:
        fig.savefig(os.path.join(working_dir, fname))
    except Exception as e:
        print(f"Error saving {fname}: {e}")
    finally:
        plt.close(fig)


# ------------------- iterate and plot ------------------- #
for exp_name, datasets in experiment_data.items():
    for dset_name, d in datasets.items():
        # unpack
        losses_tr = (
            np.array(d["losses"]["train"]) if d["losses"]["train"] else np.empty((0, 2))
        )
        losses_val = np.array(d["losses"]["val"])
        metrics_val = np.array(d["metrics"]["val"])  # (epoch,CWA,SWA,HCSA,SNWA)

        # 1) loss curve
        try:
            fig = plt.figure()
            if losses_tr.size:
                plt.plot(losses_tr[:, 0], losses_tr[:, 1], label="train")
            plt.plot(losses_val[:, 0], losses_val[:, 1], label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset_name} Loss Curves ({exp_name})")
            plt.legend()
            save_plot(fig, f"{exp_name}_{dset_name}_loss.png")
        except Exception as e:
            print(f"Error creating loss plot: {e}")
            plt.close()

        # helper for metric plots
        metric_names = ["CWA", "SWA", "HCSA", "SNWA"]
        for idx, mname in enumerate(metric_names, start=1):
            try:
                fig = plt.figure()
                plt.plot(metrics_val[:, 0], metrics_val[:, idx], label=f"val {mname}")
                plt.xlabel("Epoch")
                plt.ylabel(mname)
                plt.title(f"{dset_name} {mname} Curve ({exp_name})")
                plt.legend()
                save_plot(fig, f"{exp_name}_{dset_name}_{mname}.png")
            except Exception as e:
                print(f"Error creating {mname} plot: {e}")
                plt.close()

        # ------------------- final accuracies ------------------- #
        for split in ["dev", "test"]:
            try:
                preds = np.array(d["predictions"][split])
                gts = np.array(d["ground_truth"][split])
                acc = (preds == gts).mean() if len(gts) else float("nan")
                print(f"{exp_name}-{dset_name} {split} accuracy: {acc:.3f}")
            except Exception as e:
                print(f"Error computing accuracy for {split}: {e}")
