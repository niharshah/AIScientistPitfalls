import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
plots_done = []
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# narrow the dict we expect
run = experiment_data.get("NoClusterEmbedding", {}).get("SPR_BENCH", {})

losses = run.get("losses", {})
metrics = run.get("metrics", {})

# ---------- 1. Loss curves ----------
try:
    train_loss = losses.get("train", [])
    val_loss = losses.get("val", [])
    if train_loss or val_loss:
        epochs = range(1, max(len(train_loss), len(val_loss)) + 1)
        plt.figure()
        if train_loss:
            plt.plot(epochs, train_loss, label="Train")
        if val_loss:
            plt.plot(epochs, val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")
        plt.title("SPR_BENCH Loss Curves – Train vs Val")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
        plt.savefig(fname)
        plots_done.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- 2. Validation metric curves ----------
try:
    val_metrics = metrics.get("val", [])
    if val_metrics:
        cwa = [m.get("CWA", np.nan) for m in val_metrics]
        swa = [m.get("SWA", np.nan) for m in val_metrics]
        gcwa = [m.get("GCWA", np.nan) for m in val_metrics]
        epochs = range(1, len(val_metrics) + 1)
        plt.figure()
        plt.plot(epochs, cwa, label="CWA")
        plt.plot(epochs, swa, label="SWA")
        plt.plot(epochs, gcwa, label="GCWA")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Validation Metrics Over Epochs")
        plt.legend()
        fname = os.path.join(working_dir, "SPR_BENCH_val_metrics.png")
        plt.savefig(fname)
        plots_done.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()

# ---------- 3. Test metrics bar chart ----------
try:
    test_metrics = metrics.get("test", {})
    if test_metrics:
        labels = list(test_metrics.keys())
        values = [test_metrics[k] for k in labels]
        plt.figure()
        plt.bar(labels, values, color="skyblue")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH Test Metrics Summary")
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        fname = os.path.join(working_dir, "SPR_BENCH_test_metrics.png")
        plt.savefig(fname)
        plots_done.append(fname)
    plt.close()
except Exception as e:
    print(f"Error creating test metric plot: {e}")
    plt.close()

print("Generated plots:", plots_done)
