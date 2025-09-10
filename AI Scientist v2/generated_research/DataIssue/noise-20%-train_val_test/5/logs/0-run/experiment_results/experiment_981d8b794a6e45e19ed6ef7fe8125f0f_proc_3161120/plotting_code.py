import matplotlib.pyplot as plt
import numpy as np
import os

# set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data["num_epochs"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = None

if exp is not None:
    settings = exp["settings"]  # list of epoch counts e.g. [15,30,50]
    train_acc = exp["metrics"]["train_acc"]  # list-of-lists
    val_acc = exp["metrics"]["val_acc"]
    train_loss = exp["metrics"]["train_loss"]
    val_loss = exp["metrics"]["val_loss"]

    # small helper: longest curve length == num_epochs for that run
    def pad(l, fill=np.nan):
        m = max(map(len, l))
        return [li + [fill] * (m - len(li)) for li in l]

    # 1) accuracy curves ---------------------------------------------------
    try:
        plt.figure()
        for idx, s in enumerate(settings):
            epochs = np.arange(1, len(train_acc[idx]) + 1)
            plt.plot(epochs, train_acc[idx], label=f"Train ({s} ep)")
            plt.plot(epochs, val_acc[idx], "--", label=f"Val ({s} ep)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Accuracy Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot: {e}")
        plt.close()

    # 2) loss curves -------------------------------------------------------
    try:
        plt.figure()
        for idx, s in enumerate(settings):
            epochs = np.arange(1, len(train_loss[idx]) + 1)
            plt.plot(epochs, train_loss[idx], label=f"Train ({s} ep)")
            plt.plot(epochs, val_loss[idx], "--", label=f"Val ({s} ep)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SPR_BENCH Loss Curves\nLeft: Train, Right: Validation")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # 3) final test accuracies --------------------------------------------
    try:
        # test accuracies were printed during training; recompute quickly
        final_test_acc = []
        for preds, gts in zip(exp["predictions"], exp["ground_truth"]):
            correct = sum(int(p == g) for p, g in zip(preds, gts))
            final_test_acc.append(correct / len(gts))
        plt.figure()
        plt.bar([str(s) for s in settings], final_test_acc, color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.xlabel("Num Epochs")
        plt.title("SPR_BENCH Final Test Accuracy per Epoch Setting")
        for i, acc in enumerate(final_test_acc):
            plt.text(i, acc + 0.01, f"{acc:.2f}", ha="center")
        fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test-accuracy plot: {e}")
        plt.close()

    # print evaluation metric table
    print("Final Test Accuracies:")
    for s, acc in zip(settings, final_test_acc):
        print(f"  {s:>3d} epochs : {acc:.4f}")
