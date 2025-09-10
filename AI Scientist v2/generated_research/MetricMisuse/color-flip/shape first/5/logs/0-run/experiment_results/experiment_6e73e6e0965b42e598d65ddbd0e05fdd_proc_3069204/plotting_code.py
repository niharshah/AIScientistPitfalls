import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load experiment results
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    sweep = experiment_data["contrastive_pretraining_epochs"]["SPR_BENCH"]
    epochs = sweep["epochs"]  # list of pre-training epoch counts
    dev_hsca = sweep["metrics"]["train"]  # dev HSCA recorded after fine-tune
    test_hsca = sweep["metrics"]["val"]  # test HSCA
    train_losses = sweep["losses"]["train"]  # mean supervised loss

    print("epochs      :", epochs)
    print("dev_hsca    :", dev_hsca)
    print("test_hsca   :", test_hsca)
    print("train_losses:", train_losses)

    # --------------------------------------------------------------
    # 1) Dev HSCA vs pre-training epochs
    try:
        plt.figure()
        plt.plot(epochs, dev_hsca, marker="o")
        plt.title("SPR_BENCH – Dev HSCA vs Contrastive Pre-training Epochs")
        plt.xlabel("Contrastive pre-training epochs")
        plt.ylabel("Dev HSCA")
        plt.grid(True)
        fn = os.path.join(working_dir, "SPR_BENCH_dev_hsca_vs_pretrain_epochs.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating Dev HSCA plot: {e}")
        plt.close()

    # --------------------------------------------------------------
    # 2) Test HSCA vs pre-training epochs
    try:
        plt.figure()
        plt.plot(epochs, test_hsca, marker="s", color="orange")
        plt.title("SPR_BENCH – Test HSCA vs Contrastive Pre-training Epochs")
        plt.xlabel("Contrastive pre-training epochs")
        plt.ylabel("Test HSCA")
        plt.grid(True)
        fn = os.path.join(working_dir, "SPR_BENCH_test_hsca_vs_pretrain_epochs.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating Test HSCA plot: {e}")
        plt.close()

    # --------------------------------------------------------------
    # 3) Mean supervised loss vs pre-training epochs
    try:
        plt.figure()
        plt.plot(epochs, train_losses, marker="^", color="green")
        plt.title("SPR_BENCH – Mean Fine-tune Loss vs Contrastive Pre-training Epochs")
        plt.xlabel("Contrastive pre-training epochs")
        plt.ylabel("Mean CE Loss (training)")
        plt.grid(True)
        fn = os.path.join(working_dir, "SPR_BENCH_loss_vs_pretrain_epochs.png")
        plt.savefig(fn)
        plt.close()
    except Exception as e:
        print(f"Error creating Loss plot: {e}")
        plt.close()
