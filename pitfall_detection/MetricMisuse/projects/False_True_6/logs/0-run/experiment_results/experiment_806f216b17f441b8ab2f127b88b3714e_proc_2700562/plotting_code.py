import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

wd_dict = experiment_data.get("weight_decay", {})
tags = list(wd_dict.keys())[:5]  # ensure ≤5 tags

# -------------------------------------------------
# 1) one figure PER TAG: training vs validation loss
for tag in tags:
    try:
        losses = wd_dict[tag]["SPR_BENCH"]["losses"]
        train_loss = losses.get("train", [])
        val_loss = losses.get("val", [])
        epochs = range(1, len(train_loss) + 1)
        plt.figure()
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SPR_BENCH Loss Curves ({tag})")
        plt.legend()
        fname = os.path.join(working_dir, f"spr_loss_curves_{tag}.png")
        plt.tight_layout()
        plt.savefig(fname)
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating loss plot for {tag}: {e}")
    finally:
        plt.close()

# -------------------------------------------------
# 2) combined validation accuracy curves
try:
    plt.figure()
    for tag in tags:
        val_metrics = wd_dict[tag]["SPR_BENCH"]["metrics"]["val"]
        accs = [m.get("acc", 0) for m in val_metrics]
        plt.plot(range(1, len(accs) + 1), accs, label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("SPR_BENCH Validation Accuracy Across Weight-Decay Settings")
    plt.legend()
    fname = os.path.join(working_dir, "spr_val_acc_comparison.png")
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating validation accuracy plot: {e}")
finally:
    plt.close()

# -------------------------------------------------
# 3) bar chart of final test accuracy for each tag
try:
    tags_with_test = [t for t in tags if "test" in wd_dict[t]["SPR_BENCH"]["metrics"]]
    if tags_with_test:
        accs = [
            wd_dict[t]["SPR_BENCH"]["metrics"]["test"]["acc"] for t in tags_with_test
        ]
        plt.figure()
        plt.bar(tags_with_test, accs, color="steelblue")
        plt.ylim(0, 1)
        plt.ylabel("Test Accuracy")
        plt.title("SPR_BENCH Final Test Accuracy by Weight-Decay")
        plt.xticks(rotation=45)
        fname = os.path.join(working_dir, "spr_test_acc_bar.png")
        plt.tight_layout()
        plt.savefig(fname)
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating test accuracy bar chart: {e}")
finally:
    plt.close()
