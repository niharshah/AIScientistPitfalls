import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load experiment data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

rnn_results = experiment_data.get("rnn_hidden_dim", {})
hidden_dims = sorted(int(h) for h in rnn_results.keys())

final_accs = []

# ----- per hidden-dim loss curves (max 4) -----
for hid in hidden_dims:
    try:
        store = rnn_results[str(hid)]
        train_losses = store["losses"]["train"]
        val_losses = store["losses"]["val"]
        val_metrics = store["metrics"]["val"]
        epochs = range(1, len(train_losses) + 1)
        accs = [m["acc"] for m in val_metrics]
        final_accs.append(accs[-1])

        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"SPR_BENCH – Loss Curves (Hidden Dim {hid})")
        plt.legend()
        fname = f"spr_bench_loss_hidden{hid}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for hidden {hid}: {e}")
        plt.close()

# ----- summary accuracy vs hidden dim (1 plot) -----
try:
    if hidden_dims:
        plt.figure()
        plt.plot(hidden_dims, final_accs, marker="o")
        plt.xlabel("Hidden Dimension")
        plt.ylabel("Final Val Accuracy")
        plt.title("SPR_BENCH – Final Validation Accuracy vs. RNN Hidden Size")
        fname = "spr_bench_final_acc_vs_hidden_dim.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating accuracy summary plot: {e}")
    plt.close()
