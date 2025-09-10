import matplotlib.pyplot as plt
import numpy as np
import os

# mandatory working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to compute accuracy
def accuracy(gts, preds):
    gts, preds = np.asarray(gts), np.asarray(preds)
    return (gts == preds).mean() if len(gts) else 0.0


accs = {}  # store test accuracies for bar plot

# iterate over learning rates and create loss plots
for lr_str, rec_wrap in experiment_data.get("learning_rate", {}).items():
    rec = rec_wrap.get("spr_bench", {})
    train_losses = np.array(rec.get("losses", {}).get("train", []))  # (epoch, loss)
    val_losses = np.array(rec.get("losses", {}).get("val", []))
    try:
        plt.figure()
        if train_losses.size:
            plt.plot(train_losses[:, 0], train_losses[:, 1], label="Train")
        if val_losses.size:
            plt.plot(val_losses[:, 0], val_losses[:, 1], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"spr_bench Loss Curves (LR={lr_str})\nLeft: Train, Right: Validation"
        )
        plt.legend()
        fname = f"spr_bench_loss_curve_lr{lr_str.replace('.', '_')}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for lr={lr_str}: {e}")
        plt.close()
    # compute test accuracy
    preds = rec.get("predictions", [])
    gts = rec.get("ground_truth", [])
    accs[lr_str] = accuracy(gts, preds)

# print accuracies
for lr, a in accs.items():
    print(f"LR={lr}: test accuracy={a:.3f}")

# bar plot of final test accuracies (max 1 figure)
try:
    plt.figure()
    lrs = list(accs.keys())
    bars = [accs[k] for k in lrs]
    plt.bar(range(len(lrs)), bars, tick_label=lrs)
    plt.ylabel("Accuracy")
    plt.title("spr_bench Final Test Accuracy per Learning Rate")
    fname = "spr_bench_test_accuracy_bar.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy bar plot: {e}")
    plt.close()
