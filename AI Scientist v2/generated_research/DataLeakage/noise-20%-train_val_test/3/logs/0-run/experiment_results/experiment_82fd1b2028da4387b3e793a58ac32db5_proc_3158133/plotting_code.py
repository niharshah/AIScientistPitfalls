import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    spr_data = experiment_data["num_layers_tuning"]["SPR_BENCH"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    spr_data = {}


# Helper to fetch x/y lists
def collect(metric):
    return {
        nl: int_data["epochs"] if metric == "epochs" else int_data["metrics"][metric]
        for nl, int_data in spr_data.items()
    }


epochs_dict = collect("epochs")
train_loss_dict = collect("train_loss")
val_loss_dict = collect("val_loss")
val_f1_dict = collect("val_f1")
test_f1 = {nl: int_data.get("test_f1") for nl, int_data in spr_data.items()}

# 1) Train vs Val loss curves
try:
    plt.figure()
    for nl in sorted(train_loss_dict, key=int):
        ep = epochs_dict[nl]
        plt.plot(ep, train_loss_dict[nl], label=f"train L={nl}")
        plt.plot(ep, val_loss_dict[nl], "--", label=f"val L={nl}")
    plt.title("SPR_BENCH: Train vs Val Loss (all num_layers)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_train_val_loss_num_layers.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 2) Validation F1 curves
try:
    plt.figure()
    for nl in sorted(val_f1_dict, key=int):
        plt.plot(epochs_dict[nl], val_f1_dict[nl], label=f"L={nl}")
    plt.title("SPR_BENCH: Validation Macro-F1 vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    save_path = os.path.join(working_dir, "SPR_BENCH_val_f1_num_layers.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating F1 plot: {e}")
    plt.close()

# 3) Final test F1 bar chart
try:
    plt.figure()
    layers = sorted(test_f1, key=int)
    scores = [test_f1[l] for l in layers]
    plt.bar(layers, scores, color="skyblue")
    plt.title("SPR_BENCH: Test Macro-F1 by num_layers")
    plt.xlabel("num_layers")
    plt.ylabel("Test Macro-F1")
    for x, y in zip(layers, scores):
        plt.text(x, y + 0.01, f"{y:.2f}", ha="center")
    save_path = os.path.join(working_dir, "SPR_BENCH_test_f1_bar.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# Quick best-model printout
if test_f1:
    best_layer = max(test_f1, key=test_f1.get)
    print(f"BEST Test Macro-F1={test_f1[best_layer]:.4f} with num_layers={best_layer}")
