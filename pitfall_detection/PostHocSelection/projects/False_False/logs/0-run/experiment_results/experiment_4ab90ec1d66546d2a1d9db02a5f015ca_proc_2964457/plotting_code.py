import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data = experiment_data["emb_dim_tuning"]["SPR_BENCH"]
    train_f1 = np.array(data["metrics"]["train_macroF1"])
    val_f1 = np.array(data["metrics"]["val_macroF1"])
    train_ls = np.array(data["losses"]["train"])
    val_ls = np.array(data["losses"]["val"])
    emb_dims = np.array(data["hyperparams"])
    num_epochs = len(train_f1) // len(emb_dims) if len(emb_dims) else 0
    epoch_idx = np.arange(1, len(train_f1) + 1)

    # ------------------ Plot 1: F1 curves --------------------------
    try:
        plt.figure()
        plt.plot(epoch_idx, train_f1, label="Train Macro-F1")
        plt.plot(epoch_idx, val_f1, label="Val Macro-F1")
        plt.xlabel("Epoch (concatenated over runs)")
        plt.ylabel("Macro-F1")
        plt.title("SPR_BENCH Macro-F1 over Epochs\nLeft: Train, Right: Val")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_macroF1_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating F1 curve: {e}")
        plt.close()

    # ------------------ Plot 2: Loss curves ------------------------
    try:
        plt.figure()
        plt.plot(epoch_idx, train_ls, label="Train Loss")
        plt.plot(epoch_idx, val_ls, label="Val Loss")
        plt.xlabel("Epoch (concatenated over runs)")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("SPR_BENCH Loss over Epochs\nLeft: Train, Right: Val")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_loss_curve.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # -------- Plot 3: Final Val F1 vs Embedding Dimension ----------
    try:
        finals = val_f1.reshape(len(emb_dims), num_epochs)[:, -1]
        plt.figure()
        plt.bar([str(e) for e in emb_dims], finals, color="skyblue")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Final Val Macro-F1")
        plt.title("SPR_BENCH Final Validation Macro-F1 by Embedding Size")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_valF1_vs_embdim.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating emb-dim bar plot: {e}")
        plt.close()

    # ------------------ Print best run -----------------------------
    if len(emb_dims) and num_epochs:
        best_idx = finals.argmax()
        print(
            f"Best emb_dim={emb_dims[best_idx]} | Val Macro-F1={finals[best_idx]:.4f}"
        )
