import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset_name = "SPR_BENCH"
variants = list(experiment_data.keys())

# --------- collect for bar chart ----------
test_table = {}  # variant -> {metric: value}
for v in variants:
    try:
        test_table[v] = experiment_data[v][dataset_name]["metrics"]["test"]
    except KeyError:
        pass

# --------- figure 1-3: per-variant loss curves ----------
for v in variants:
    try:
        data = experiment_data[v][dataset_name]
        epochs = data["epochs"]
        t_loss = data["losses"]["train"]
        v_loss = data["losses"]["val"]

        plt.figure()
        plt.plot(epochs, t_loss, label="Train")
        plt.plot(epochs, v_loss, label="Validation")
        plt.title(f"{dataset_name}: {v} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.legend()
        fname = f"loss_{dataset_name}_{v}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {v}: {e}")
        plt.close()

# --------- figure 4: comparison of val CplxWA ----------
try:
    plt.figure()
    for v in variants:
        data = experiment_data[v][dataset_name]
        epochs = data["epochs"]
        cplx = data["metrics"]["val"]["CplxWA"]
        plt.plot(epochs, cplx, label=v)
    plt.title(f"{dataset_name}: Validation Complexity-WA")
    plt.xlabel("Epoch")
    plt.ylabel("CplxWA")
    plt.legend()
    fname = f"val_CplxWA_comparison_{dataset_name}.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating CplxWA comparison plot: {e}")
    plt.close()

# --------- figure 5: bar chart of test metrics ----------
try:
    metrics = ["CWA", "SWA", "CplxWA"]
    x = np.arange(len(metrics))
    width = 0.25
    plt.figure()
    for i, v in enumerate(variants):
        vals = [test_table[v][m] for m in metrics]
        plt.bar(x + i * width, vals, width, label=v)
    plt.title(f"{dataset_name}: Test Metrics by Variant")
    plt.xticks(x + width, metrics)
    plt.ylabel("Accuracy")
    plt.legend()
    fname = f"test_metrics_{dataset_name}.png"
    plt.savefig(os.path.join(working_dir, fname))
    plt.close()
except Exception as e:
    print(f"Error creating test metrics bar chart: {e}")
    plt.close()

# --------- print evaluation numbers ----------
for v in variants:
    print(f"{v} Test:", {k: round(vv, 3) for k, vv in test_table[v].items()})
