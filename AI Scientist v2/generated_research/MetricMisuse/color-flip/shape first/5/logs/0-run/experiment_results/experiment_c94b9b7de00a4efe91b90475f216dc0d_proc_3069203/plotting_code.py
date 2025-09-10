import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

temp_dict = experiment_data.get("contrastive_temperature", {})
temperatures = sorted([float(t) for t in temp_dict.keys()])
dev_curves, test_scores, train_losses = {}, {}, {}

for t_str, d in temp_dict.items():
    dev_curves[float(t_str)] = d["metrics"]["train"]  # Dev HSCA per epoch
    test_scores[float(t_str)] = d["metrics"]["val"][0]  # Final Test HSCA
    train_losses[float(t_str)] = d["losses"]["train"]  # Supervised train loss

best_temp = max(test_scores, key=test_scores.get) if test_scores else None

# ------------------------------------------------------------------
# 1) Dev HSCA curves
try:
    plt.figure()
    for t in temperatures:
        epochs = np.arange(1, len(dev_curves[t]) + 1)
        plt.plot(epochs, dev_curves[t], label=f"τ={t}")
    plt.xlabel("Fine-tuning Epoch")
    plt.ylabel("HSCA (Dev)")
    plt.title("SPR_BENCH Synthetic – Dev HSCA vs Epoch (Contrastive Temperature Sweep)")
    plt.legend()
    fname = os.path.join(working_dir, "sprbench_dev_hsca_vs_epoch.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating Dev HSCA plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 2) Final Test HSCA vs temperature
try:
    plt.figure()
    temps = list(test_scores.keys())
    scores = [test_scores[t] for t in temps]
    plt.bar([str(t) for t in temps], scores)
    plt.xlabel("Contrastive Temperature τ")
    plt.ylabel("HSCA (Test)")
    plt.title("SPR_BENCH Synthetic – Final Test HSCA across Temperatures")
    fname = os.path.join(working_dir, "sprbench_test_hsca_vs_temperature.png")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating Test HSCA bar plot: {e}")
    plt.close()

# ------------------------------------------------------------------
# 3) Training loss curve for best temperature
try:
    if best_temp is not None:
        plt.figure()
        epochs = np.arange(1, len(train_losses[best_temp]) + 1)
        plt.plot(epochs, train_losses[best_temp], marker="o")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"SPR_BENCH Synthetic – Training Loss vs Epoch (Best τ={best_temp})")
        fname = os.path.join(
            working_dir, f"sprbench_train_loss_best_tau_{best_temp}.png"
        )
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()
