import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# --- dirs ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- load data ---
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Helper to collect logs
layers_dict = experiment_data.get("num_lstm_layers", {}).get("SPR_BENCH", {})
layer_keys = sorted(layers_dict.keys(), key=int)  # ['1','2','3']

# ---- 1) F1 curves ----
try:
    plt.figure()
    for k in layer_keys:
        epochs = layers_dict[k]["epochs"]
        plt.plot(epochs, layers_dict[k]["metrics"]["train"], label=f"train_L{k}")
        plt.plot(
            epochs, layers_dict[k]["metrics"]["val"], label=f"val_L{k}", linestyle="--"
        )
    plt.title("SPR_BENCH: Macro F1 vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_F1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# ---- 2) Loss curves ----
try:
    plt.figure()
    for k in layer_keys:
        epochs = layers_dict[k]["epochs"]
        plt.plot(epochs, layers_dict[k]["losses"]["train"], label=f"train_L{k}")
        plt.plot(
            epochs, layers_dict[k]["losses"]["val"], label=f"val_L{k}", linestyle="--"
        )
    plt.title("SPR_BENCH: Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_Loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Loss curve plot: {e}")
    plt.close()

# ---- 3) Bar chart of best Dev/Test F1 ----
try:
    best_dev = [layers_dict[k]["best_dev_f1"] for k in layer_keys]
    test_f1 = [layers_dict[k]["test_f1"] for k in layer_keys]
    x = np.arange(len(layer_keys))
    width = 0.35
    plt.figure()
    plt.bar(x - width / 2, best_dev, width, label="Best Dev F1")
    plt.bar(x + width / 2, test_f1, width, label="Test F1")
    plt.xticks(x, [f"L{k}" for k in layer_keys])
    plt.ylabel("Macro F1")
    plt.title("SPR_BENCH: Dev vs Test F1 by #LSTM layers")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "SPR_BENCH_Dev_Test_F1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# ---- 4-6) Confusion matrices (one per layer) ----
for k in layer_keys:
    try:
        preds = layers_dict[k]["predictions"]
        gts = layers_dict[k]["ground_truth"]
        if preds is None or gts is None or len(preds) == 0:
            continue
        cm = confusion_matrix(gts, preds)
        plt.figure()
        plt.imshow(cm, cmap="Blues")
        plt.title(f"SPR_BENCH Confusion Matrix - LSTM Layers={k}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.colorbar()
        fname = f"SPR_BENCH_confusion_matrix_L{k}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for L{k}: {e}")
        plt.close()

# ---- print summary table ----
print("Layer | Best Dev F1 | Test F1")
for k in layer_keys:
    print(
        f"  {k}   |  {layers_dict[k]['best_dev_f1']:.4f}   | {layers_dict[k]['test_f1']:.4f}"
    )
