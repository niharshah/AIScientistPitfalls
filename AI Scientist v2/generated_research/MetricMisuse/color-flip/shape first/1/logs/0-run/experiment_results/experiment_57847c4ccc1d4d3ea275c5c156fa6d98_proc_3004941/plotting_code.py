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

if experiment_data is not None:
    data = experiment_data["SPR_BENCH"]
    # 1) Pre-training loss
    try:
        plt.figure()
        plt.plot(
            range(1, len(data["losses"]["pretrain"]) + 1),
            data["losses"]["pretrain"],
            label="pretrain loss",
        )
        plt.title("SPR_BENCH – Pre-training Loss")
        plt.xlabel("Self-supervised epoch")
        plt.ylabel("NT-Xent loss")
        plt.legend()
        fn = os.path.join(working_dir, "SPR_BENCH_pretrain_loss.png")
        plt.savefig(fn)
        plt.close()
        print(f"Saved {fn}")
    except Exception as e:
        print(f"Error creating pretraining plot: {e}")
        plt.close()

    # 2) Fine-tune train / val loss
    try:
        plt.figure()
        x = range(1, len(data["losses"]["train"]) + 1)
        plt.plot(x, data["losses"]["train"], label="train")
        plt.plot(x, data["losses"]["val"], label="val")
        plt.title("SPR_BENCH – Fine-tune Loss")
        plt.xlabel("Fine-tune step (sequential)")
        plt.ylabel("Cross-entropy loss")
        plt.legend()
        fn = os.path.join(working_dir, "SPR_BENCH_finetune_train_val_loss.png")
        plt.savefig(fn)
        plt.close()
        print(f"Saved {fn}")
    except Exception as e:
        print(f"Error creating fine-tune loss plot: {e}")
        plt.close()

    # 3) Weighted accuracies
    try:
        plt.figure()
        x = range(1, len(data["metrics"]["SWA"]) + 1)
        plt.plot(x, data["metrics"]["SWA"], label="SWA")
        plt.plot(x, data["metrics"]["CWA"], label="CWA")
        plt.plot(x, data["metrics"]["SCWA"], label="SCWA")
        plt.title("SPR_BENCH – Weighted Accuracies")
        plt.xlabel("Fine-tune step (sequential)")
        plt.ylabel("Accuracy")
        plt.legend()
        fn = os.path.join(working_dir, "SPR_BENCH_weighted_accuracies.png")
        plt.savefig(fn)
        plt.close()
        print(f"Saved {fn}")
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()
