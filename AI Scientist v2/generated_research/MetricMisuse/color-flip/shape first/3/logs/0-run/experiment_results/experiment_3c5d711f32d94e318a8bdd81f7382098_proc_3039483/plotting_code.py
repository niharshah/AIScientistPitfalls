import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------------- #
# Setup & load data
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()["multi_synth_generalization"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data:
    datasets = ["variety", "length_parity", "majority_shape"]

    # 1-3) per-dataset loss curves ---------------------------------------- #
    for ds in datasets:
        try:
            tr_loss = experiment_data[ds]["losses"]["train"]
            val_loss = experiment_data[ds]["losses"]["val"]
            epochs = range(1, len(tr_loss) + 1)

            plt.figure()
            plt.plot(epochs, tr_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.title(f"{ds} dataset – Training vs Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_loss_curves.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error plotting loss for {ds}: {e}")
            plt.close()

    # 4) aggregate validation accuracy curves ----------------------------- #
    try:
        plt.figure()
        for ds in datasets:
            val_acc = experiment_data[ds]["metrics"]["val"]  # accuracy stored here
            plt.plot(range(1, len(val_acc) + 1), val_acc, label=ds)
        plt.title("Validation Accuracy Across Datasets")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.ylim(0, 1)
        fname = os.path.join(working_dir, "val_accuracy_all_datasets.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error plotting aggregated accuracy: {e}")
        plt.close()

    # 5) transfer accuracy heat-map --------------------------------------- #
    try:
        mat = np.zeros((3, 3))
        for i, src in enumerate(datasets):
            # diagonal: final validation accuracy
            mat[i, i] = experiment_data[src]["metrics"]["val"][-1]
            for j, tgt in enumerate(datasets):
                if i == j:
                    continue
                key = f"{src}_to_{tgt}"
                mat[i, j] = experiment_data["transfer"][key]["acc"]
        plt.figure()
        im = plt.imshow(mat, vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(im)
        plt.xticks(range(3), datasets, rotation=45)
        plt.yticks(range(3), datasets)
        plt.title(
            "Left/Top: Source Dataset, Right/Bottom: Target Dataset\nTransfer Accuracy Heat-map"
        )
        for i in range(3):
            for j in range(3):
                plt.text(
                    j,
                    i,
                    f"{mat[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="w" if mat[i, j] < 0.5 else "k",
                )
        fname = os.path.join(working_dir, "transfer_accuracy_heatmap.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error plotting transfer heatmap: {e}")
        plt.close()

    # --------------------------------------------------------------------- #
    # Print final metrics
    final_val_acc = {ds: experiment_data[ds]["metrics"]["val"][-1] for ds in datasets}
    print("Final Validation Accuracies:", final_val_acc)
    print("Transfer Accuracy Matrix (rows=source, cols=target):\n", mat)
