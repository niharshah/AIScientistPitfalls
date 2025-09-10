import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# ---------------------------------------------------------------------
# Iterate through experiments and create plots
for exp_name, datasets in experiment_data.items():
    for dset_name, info in datasets.items():
        # 1) Loss curves -------------------------------------------------
        try:
            train_loss = np.array(info["losses"]["train"])  # (epoch, value)
            val_loss = np.array(info["losses"]["val"])
            if train_loss.size and val_loss.size:
                plt.figure()
                plt.plot(train_loss[:, 0], train_loss[:, 1], label="Train")
                plt.plot(val_loss[:, 0], val_loss[:, 1], label="Validation")
                plt.xlabel("Epoch")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(f"{exp_name}-{dset_name}: Loss vs Epochs")
                plt.legend()
                fname = f"{exp_name}_{dset_name}_loss_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating loss plot for {exp_name}-{dset_name}: {e}")
            plt.close()

        # 2) PCWA metric curves -----------------------------------------
        try:
            train_pcwa = np.array(info["metrics"]["train"])
            val_pcwa = np.array(info["metrics"]["val"])
            if train_pcwa.size and val_pcwa.size:
                plt.figure()
                plt.plot(train_pcwa[:, 0], train_pcwa[:, 1], label="Train PCWA")
                plt.plot(val_pcwa[:, 0], val_pcwa[:, 1], label="Val PCWA")
                plt.xlabel("Epoch")
                plt.ylabel("PCWA")
                plt.title(f"{exp_name}-{dset_name}: PCWA vs Epochs")
                plt.legend()
                fname = f"{exp_name}_{dset_name}_pcwa_curve.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating PCWA plot for {exp_name}-{dset_name}: {e}")
            plt.close()

        # 3) Confusion matrix on test set --------------------------------
        try:
            y_true = np.array(info.get("ground_truth", []))
            y_pred = np.array(info.get("predictions", []))
            if y_true.size and y_pred.size:
                conf = np.zeros((2, 2), dtype=int)
                for t, p in zip(y_true, y_pred):
                    conf[t, p] += 1
                plt.figure()
                plt.imshow(conf, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        plt.text(
                            j, i, conf[i, j], ha="center", va="center", color="black"
                        )
                plt.xticks([0, 1], ["Pred 0", "Pred 1"])
                plt.yticks([0, 1], ["True 0", "True 1"])
                plt.title(f"{exp_name}-{dset_name}: Confusion Matrix")
                plt.colorbar()
                fname = f"{exp_name}_{dset_name}_confusion_matrix.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {exp_name}-{dset_name}: {e}")
            plt.close()

        # 4) Print test metrics if available -----------------------------
        metrics = info.get("metrics", {}).get("test") or info.get("test_metrics")
        if metrics:
            print(f"{exp_name}-{dset_name} TEST:", metrics)
