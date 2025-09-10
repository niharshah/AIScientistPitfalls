import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Helper: gather keys and colours --------------------------------------------------------
keys = list(experiment_data.get("num_epochs", {}).keys())
colors = plt.cm.tab10.colors if keys else []

# 1) Train / Val Macro-F1 curves ----------------------------------------------------------
try:
    plt.figure()
    for idx, k in enumerate(keys):
        epochs = experiment_data["num_epochs"][k]["epochs"]
        tr_f1 = experiment_data["num_epochs"][k]["metrics"]["train_macro_f1"]
        val_f1 = experiment_data["num_epochs"][k]["metrics"]["val_macro_f1"]
        c = colors[idx % len(colors)]
        plt.plot(epochs, tr_f1, linestyle="--", color=c, label=f"{k}-train")
        plt.plot(epochs, val_f1, linestyle="-", color=c, label=f"{k}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curves (Left: Train dashed, Right: Validation solid)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_macro_f1_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating macro-F1 plot: {e}")
    plt.close()

# 2) Train / Val Loss curves --------------------------------------------------------------
try:
    plt.figure()
    for idx, k in enumerate(keys):
        epochs = experiment_data["num_epochs"][k]["epochs"]
        tr_loss = experiment_data["num_epochs"][k]["losses"]["train"]
        val_loss = experiment_data["num_epochs"][k]["losses"]["val"]
        c = colors[idx % len(colors)]
        plt.plot(epochs, tr_loss, linestyle="--", color=c, label=f"{k}-train")
        plt.plot(epochs, val_loss, linestyle="-", color=c, label=f"{k}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves (Left: Train dashed, Right: Validation solid)")
    plt.legend()
    fname = os.path.join(working_dir, "spr_bench_loss_curves.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# 3) Test Macro-F1 bar chart --------------------------------------------------------------
test_scores = {}
try:
    for k in keys:
        test_scores[k] = experiment_data["num_epochs"][k].get("test_macro_f1", np.nan)

    plt.figure()
    plt.bar(
        range(len(test_scores)),
        list(test_scores.values()),
        tick_label=list(test_scores.keys()),
    )
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Test Macro-F1 per Hyper-param Setting")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = os.path.join(working_dir, "spr_bench_test_macro_f1_bar.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test score bar plot: {e}")
    plt.close()

# Print numeric summary -------------------------------------------------------------------
print("Test Macro-F1 scores:", test_scores)
