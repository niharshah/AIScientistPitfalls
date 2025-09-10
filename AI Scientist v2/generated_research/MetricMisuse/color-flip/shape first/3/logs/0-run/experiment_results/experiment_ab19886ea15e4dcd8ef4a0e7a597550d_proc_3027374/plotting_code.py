import matplotlib.pyplot as plt
import numpy as np
import os

# working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None and "SPR_BENCH" in experiment_data:
    ds = experiment_data["SPR_BENCH"]
    # handy aliases
    pre_losses = ds["losses"].get("pretrain", [])
    ft_losses = ds["losses"].get("finetune", [])
    val_loss = ds["metrics"].get("val_loss", [])
    val_swa = ds["metrics"].get("val_SWA", [])
    val_aca = ds["metrics"].get("val_ACA", [])
    preds = np.array(ds.get("predictions", []))
    gts = np.array(ds.get("ground_truth", []))

    # ---------------------------------------------------- Plot 1: pre-training loss
    try:
        if len(pre_losses):
            plt.figure()
            plt.plot(range(1, len(pre_losses) + 1), pre_losses, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Contrastive Loss")
            plt.title("SPR_BENCH: Pre-training Contrastive Loss vs Epoch")
            fn = os.path.join(working_dir, "SPR_BENCH_pretrain_loss.png")
            plt.savefig(fn)
            plt.close()
            print(f"Saved {fn}")
    except Exception as e:
        print(f"Error creating pretrain loss plot: {e}")
        plt.close()

    # ---------------------------------------------------- Plot 2: fine-tune train loss
    try:
        if len(ft_losses):
            plt.figure()
            plt.plot(range(1, len(ft_losses) + 1), ft_losses, marker="o", color="g")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-entropy Loss")
            plt.title("SPR_BENCH: Fine-tuning Training Loss vs Epoch")
            fn = os.path.join(working_dir, "SPR_BENCH_finetune_train_loss.png")
            plt.savefig(fn)
            plt.close()
            print(f"Saved {fn}")
    except Exception as e:
        print(f"Error creating finetune loss plot: {e}")
        plt.close()

    # ---------------------------------------------------- Plot 3: validation SWA & ACA
    try:
        if len(val_swa) and len(val_aca):
            epochs = range(1, len(val_swa) + 1)
            plt.figure()
            plt.plot(epochs, val_swa, label="SWA", marker="o")
            plt.plot(epochs, val_aca, label="ACA", marker="s")
            plt.xlabel("Epoch")
            plt.ylabel("Score")
            plt.title("SPR_BENCH: Validation SWA & ACA vs Epoch")
            plt.legend()
            fn = os.path.join(working_dir, "SPR_BENCH_val_swa_aca.png")
            plt.savefig(fn)
            plt.close()
            print(f"Saved {fn}")
    except Exception as e:
        print(f"Error creating val metric plot: {e}")
        plt.close()

    # ---------------------------------------------------- Plot 4: validation loss
    try:
        if len(val_loss):
            plt.figure()
            plt.plot(range(1, len(val_loss) + 1), val_loss, color="r", marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title("SPR_BENCH: Validation Loss vs Epoch")
            fn = os.path.join(working_dir, "SPR_BENCH_val_loss.png")
            plt.savefig(fn)
            plt.close()
            print(f"Saved {fn}")
    except Exception as e:
        print(f"Error creating val loss plot: {e}")
        plt.close()

    # ---------------------------------------------------- Plot 5: confusion matrix
    try:
        if preds.size and gts.size:
            from itertools import product

            cm = np.zeros((2, 2), dtype=int)
            for t, p in zip(gts, preds):
                cm[t, p] += 1
            plt.figure()
            plt.imshow(cm, cmap="Blues")
            for i, j in product(range(2), range(2)):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])
            plt.title("SPR_BENCH: Confusion Matrix (Test Set)")
            fn = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
            plt.savefig(fn)
            plt.close()
            print(f"Saved {fn}")
    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")
        plt.close()

    # ---------------------------------------------------- Print final test metrics
    if preds.size and gts.size:
        swa_final = (
            ds["metrics"]["val_SWA"][-1] if len(ds["metrics"]["val_SWA"]) else None
        )
        aca_final = (
            ds["metrics"]["val_ACA"][-1] if len(ds["metrics"]["val_ACA"]) else None
        )
        print(f"Final validation SWA: {swa_final}, ACA: {aca_final}")
else:
    print("SPR_BENCH data not found in experiment_data.npy")
