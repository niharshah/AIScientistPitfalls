import matplotlib.pyplot as plt
import numpy as np
import os

# prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data ---------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

# iterate over all datasets in the npy dict ------------------------------------
for dset, bench in experiment_data.items():
    # helper epoch indices (fine-tuning epochs)
    epochs_ft = np.arange(1, len(bench["losses"]["train"]) + 1)

    # 1) contrastive + fine-tune loss curves -----------------------------------
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        # left: contrastive loss
        if bench["losses"]["contrastive"]:
            ax[0].plot(
                np.arange(1, len(bench["losses"]["contrastive"]) + 1),
                bench["losses"]["contrastive"],
                label="Contrastive Loss",
            )
            ax[0].set_title(f"{dset}: Contrastive Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
        else:
            ax[0].text(0.5, 0.5, "No contrastive\nloss logged", ha="center")
            ax[0].set_axis_off()

        # right: supervised train/val loss
        ax[1].plot(epochs_ft, bench["losses"]["train"], label="Train")
        ax[1].plot(epochs_ft, bench["losses"]["val"], label="Validation")
        ax[1].set_title(f"{dset}: CE Loss (Fine-tune)")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")
        ax[1].legend()

        fname = os.path.join(working_dir, f"{dset.lower()}_loss_curves.png")
        plt.tight_layout()
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dset}: {e}")
        plt.close()

    # 2) metric curves ---------------------------------------------------------
    try:
        plt.figure()
        for mkey, lbl in zip(["SWA", "CWA", "SCWA"], ["SWA", "CWA", "SCWA"]):
            if bench["metrics"][mkey]:
                plt.plot(epochs_ft, bench["metrics"][mkey], label=lbl)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{dset}: Weighted Accuracy Metrics")
        plt.legend()
        fname = os.path.join(working_dir, f"{dset.lower()}_metric_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {dset}: {e}")
        plt.close()

    # 3) confusion matrix ------------------------------------------------------
    try:
        gt, pr = bench["ground_truth"], bench["predictions"]
        if gt and pr:
            labels = sorted({*gt, *pr})
            lab2idx = {lab: i for i, lab in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=int)
            for g, p in zip(gt, pr):
                cm[lab2idx[g], lab2idx[p]] += 1

            plt.figure(figsize=(5, 4))
            im = plt.imshow(cm, cmap="Blues")
            plt.colorbar(im, fraction=0.046)
            plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Predicted")
            plt.ylabel("Ground Truth")
            plt.title(f"{dset}: Confusion Matrix (Last Epoch)")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fname = os.path.join(working_dir, f"{dset.lower()}_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(fname)
            print("Saved", fname)
            plt.close()
        else:
            print(f"No ground-truth/prediction data for {dset}, skipping CM plot.")
    except Exception as e:
        print(f"Error creating confusion matrix for {dset}: {e}")
        plt.close()
