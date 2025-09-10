import matplotlib.pyplot as plt
import numpy as np
import os

# set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper for safe extraction
def split_xy(pairs):
    xs, ys = zip(*pairs) if pairs else ([], [])
    return list(xs), list(ys)


# iterate over experiments and datasets
for exp_name, datasets in experiment_data.items():
    for dname, content in datasets.items():
        # ------------------ LOSS CURVES ------------------ #
        try:
            plt.figure()
            # training losses
            x_train, y_train = split_xy(content.get("losses", {}).get("train", []))
            if x_train:
                plt.plot(x_train, y_train, label="train")
            # validation losses
            x_val, y_val = split_xy(content.get("losses", {}).get("val", []))
            if x_val:
                plt.plot(x_val, y_val, label="val")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"{dname} Loss Curves ({exp_name})")
            plt.legend()
            fname = f"{dname}_{exp_name}_loss_curves.png".replace(" ", "_")
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {dname}: {e}")
            plt.close()

        # ------------------ METRIC CURVES ---------------- #
        try:
            plt.figure()
            metrics = content.get("metrics", {}).get("val", [])
            if metrics:
                epochs, cwa, swa, hcs, snwa = zip(*metrics)
                plt.plot(epochs, cwa, label="CWA")
                plt.plot(epochs, swa, label="SWA")
                plt.plot(epochs, hcs, label="HCSA")
                plt.plot(epochs, snwa, label="SNWA")
            plt.xlabel("epoch")
            plt.ylabel("score")
            plt.title(f"{dname} Validation Metrics ({exp_name})")
            plt.legend()
            fname = f"{dname}_{exp_name}_val_metrics.png".replace(" ", "_")
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating metric plot for {dname}: {e}")
            plt.close()

        # --------------- PRINT FINAL METRICS -------------- #
        if content.get("metrics", {}).get("val"):
            last = content["metrics"]["val"][-1]
            print(
                f"{dname} final metrics (epoch {last[0]}): "
                f"CWA={last[1]:.3f}, SWA={last[2]:.3f}, "
                f"HCSA={last[3]:.3f}, SNWA={last[4]:.3f}"
            )
