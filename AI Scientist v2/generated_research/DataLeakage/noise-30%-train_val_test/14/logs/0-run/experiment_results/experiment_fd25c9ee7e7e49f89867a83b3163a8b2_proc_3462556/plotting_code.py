import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dct = experiment_data.get("d_model_tuning", {})
dataset_name = "SPR_BENCH"

# gather epochs and metrics
d_models = sorted(dct.keys(), key=int)
epochs = {}
loss_tr, loss_val, f1_tr, f1_val, final_val_f1 = {}, {}, {}, {}, {}
for dm in d_models:
    run = dct[dm][dataset_name]
    loss_tr[dm] = [x["loss"] for x in run["losses"]["train"]]
    loss_val[dm] = [x["loss"] for x in run["losses"]["val"]]
    f1_tr[dm] = [x["macro_f1"] for x in run["metrics"]["train"]]
    f1_val[dm] = [x["macro_f1"] for x in run["metrics"]["val"]]
    epochs[dm] = [x["epoch"] for x in run["metrics"]["train"]]
    final_val_f1[dm] = f1_val[dm][-1] if f1_val[dm] else 0.0

# 1) Loss curves
try:
    plt.figure()
    for dm in d_models:
        plt.plot(epochs[dm], loss_tr[dm], label=f"{dm}-train", linestyle="--")
        plt.plot(epochs[dm], loss_val[dm], label=f"{dm}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training vs Validation Loss\nDataset: SPR_BENCH")
    plt.legend()
    save_name = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(save_name)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves: {e}")
    plt.close()

# 2) Macro-F1 curves
try:
    plt.figure()
    for dm in d_models:
        plt.plot(epochs[dm], f1_tr[dm], label=f"{dm}-train", linestyle="--")
        plt.plot(epochs[dm], f1_val[dm], label=f"{dm}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training vs Validation Macro-F1\nDataset: SPR_BENCH")
    plt.legend()
    save_name = os.path.join(working_dir, "SPR_BENCH_f1_curves.png")
    plt.savefig(save_name)
    plt.close()
except Exception as e:
    print(f"Error creating F1 curves: {e}")
    plt.close()

# 3) Bar chart of final val F1
try:
    plt.figure()
    xs = np.arange(len(d_models))
    vals = [final_val_f1[dm] for dm in d_models]
    plt.bar(xs, vals, tick_label=d_models)
    plt.xlabel("d_model")
    plt.ylabel("Final Val Macro-F1")
    plt.title("Final Validation Macro-F1 per d_model\nDataset: SPR_BENCH")
    save_name = os.path.join(working_dir, "SPR_BENCH_final_val_f1_bar.png")
    plt.savefig(save_name)
    plt.close()
except Exception as e:
    print(f"Error creating bar chart: {e}")
    plt.close()

# ------------------------------------------------------------------
# Print best d_model
if final_val_f1:
    best_dm = max(final_val_f1, key=final_val_f1.get)
    print(
        f"Best d_model by final validation Macro-F1: {best_dm} "
        f"({final_val_f1[best_dm]:.4f})"
    )
