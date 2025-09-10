import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------- load experiment data --------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

drop_dict = experiment_data.get("dropout_rate", {})

# -------- plot loss curves for each dropout --------
for dp in sorted([k for k in drop_dict.keys() if isinstance(k, float)]):
    try:
        hist = drop_dict[dp]["SPR_BENCH"]
        train_losses = hist["losses"]["train"]
        val_losses = hist["losses"]["val"]
        epochs = np.arange(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Train vs Validation Loss\nDropout = {dp}")
        plt.legend()
        fname = f"loss_curve_SPR_BENCH_dropout_{dp}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for dropout {dp}: {e}")
        plt.close()

# -------- print best model results --------
best_rate = drop_dict.get("best_rate", None)
test_metrics = drop_dict.get("test_metrics", {})
print(f"Best dropout rate: {best_rate}")
print("Test metrics:", test_metrics)
