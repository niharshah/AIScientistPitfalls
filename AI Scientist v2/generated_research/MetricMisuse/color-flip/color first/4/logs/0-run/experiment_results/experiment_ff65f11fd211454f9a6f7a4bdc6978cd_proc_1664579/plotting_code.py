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
    experiment_data = {}

rec = experiment_data.get("SPR_BENCH", {})
loss_tr = rec.get("losses", {}).get("train", [])
loss_val = rec.get("losses", {}).get("val", [])
metrics = rec.get("metrics", {}).get("val", [])

acc = [m["acc"] for m in metrics] if metrics else []
cwa = [m["CWA"] for m in metrics] if metrics else []
swa = [m["SWA"] for m in metrics] if metrics else []
comp = [m["CompWA"] for m in metrics] if metrics else []

epochs = list(range(1, len(loss_tr) + 1))


def plot_line(y_lists, labels, ylab, fname):
    try:
        plt.figure()
        for y, lab in zip(y_lists, labels):
            if y:
                plt.plot(epochs, y, label=lab)
        plt.title(f"SPR_BENCH {ylab} over Epochs\n(Line plot)")
        plt.xlabel("Epoch")
        plt.ylabel(ylab)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"SPR_BENCH_{fname}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot {fname}: {e}")
        plt.close()


plot_line([loss_tr, loss_val], ["Train", "Validation"], "Loss", "loss_curves")
plot_line([acc], ["Validation ACC"], "Accuracy", "val_accuracy")
plot_line([cwa], ["CWA"], "Color-Weighted Accuracy", "cwa")
plot_line([swa], ["SWA"], "Shape-Weighted Accuracy", "swa")
plot_line([comp], ["CompWA"], "Composite-Weighted Accuracy", "compwa")

if epochs:
    print(
        f"Final epoch ({epochs[-1]}): "
        f"Loss={loss_val[-1]:.4f}, ACC={acc[-1]:.3f}, "
        f"CWA={cwa[-1]:.3f}, SWA={swa[-1]:.3f}, CompWA={comp[-1]:.3f}"
    )
