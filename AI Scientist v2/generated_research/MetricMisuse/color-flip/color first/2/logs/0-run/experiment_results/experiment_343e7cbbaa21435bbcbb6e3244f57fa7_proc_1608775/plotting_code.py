import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

for dset, dset_dict in experiment_data.items():  # e.g. SPR_BENCH
    for model, md in dset_dict.items():  # e.g. transformer_cnn
        train_loss = md["losses"].get("train", [])
        val_loss = md["losses"].get("val", [])
        val_metrics = md["metrics"].get("val", [])
        test_metrics = md["metrics"].get("test", {})
        epochs = range(1, len(train_loss) + 1)

        # -------- plot 1: training loss ----------
        try:
            plt.figure()
            plt.plot(epochs, train_loss, label="train")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset} {model} Training Loss vs Epoch")
            plt.legend()
            fname = f"{dset}_{model}_training_loss.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error training loss plot ({dset}/{model}): {e}")
            plt.close()

        # -------- plot 2: validation loss ----------
        try:
            plt.figure()
            plt.plot(epochs, val_loss, label="val", color="orange")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset} {model} Validation Loss vs Epoch")
            plt.legend()
            fname = f"{dset}_{model}_validation_loss.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error validation loss plot ({dset}/{model}): {e}")
            plt.close()

        # -------- plot 3: validation metrics ----------
        try:
            if val_metrics:
                cwa = [m["CWA"] for m in val_metrics]
                swa = [m["SWA"] for m in val_metrics]
                gcwa = [m["GCWA"] for m in val_metrics]
                plt.figure()
                plt.plot(epochs, cwa, label="CWA")
                plt.plot(epochs, swa, label="SWA")
                plt.plot(epochs, gcwa, label="GCWA")
                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.ylim(0, 1)
                plt.title(
                    f"{dset} {model} Validation Metrics vs Epoch\n"
                    "Left: CWA, Center: SWA, Right: GCWA"
                )
                plt.legend()
                fname = f"{dset}_{model}_validation_metrics.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error validation metrics plot ({dset}/{model}): {e}")
            plt.close()

        # -------- plot 4: test metrics ----------
        try:
            if test_metrics:
                labels = ["CWA", "SWA", "GCWA"]
                vals = [test_metrics.get(k, 0) for k in labels]
                x = np.arange(len(labels))
                plt.figure()
                plt.bar(x, vals, color=["steelblue", "orange", "green"])
                plt.xticks(x, labels)
                plt.ylim(0, 1)
                plt.ylabel("Score")
                plt.title(f"{dset} {model} Final Test Metrics")
                fname = f"{dset}_{model}_test_metrics.png"
                plt.savefig(os.path.join(working_dir, fname))
                plt.close()
        except Exception as e:
            print(f"Error test metrics plot ({dset}/{model}): {e}")
            plt.close()

        # -------- print metrics ----------
        if test_metrics:
            print(f"{dset} / {model} test metrics:")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.3f}")
