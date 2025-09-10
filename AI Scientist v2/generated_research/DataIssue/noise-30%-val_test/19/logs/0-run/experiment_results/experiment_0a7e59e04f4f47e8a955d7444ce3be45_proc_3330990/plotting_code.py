import matplotlib.pyplot as plt
import numpy as np
import os

# ------------ paths & data loading -------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    emb_logs = experiment_data.get("emb_dim", {})
    # ------- per-embedding plots (max 4 -> still <=5) -------
    for emb_dim, logs in emb_logs.items():
        try:
            epochs = logs.get("epochs", [])
            tr_loss = logs.get("losses", {}).get("train", [])
            val_loss = logs.get("losses", {}).get("val", [])
            tr_f1 = logs.get("metrics", {}).get("train_macro_f1", [])
            val_f1 = logs.get("metrics", {}).get("val_macro_f1", [])

            plt.figure(figsize=(10, 4))
            # Left: Loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, tr_loss, label="Train")
            plt.plot(epochs, val_loss, label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss")
            plt.legend()

            # Right: Macro-F1
            plt.subplot(1, 2, 2)
            plt.plot(epochs, tr_f1, label="Train F1")
            plt.plot(epochs, val_f1, label="Val F1")
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title("Macro-F1")
            plt.legend()

            plt.suptitle(
                f"Left: Loss, Right: Macro-F1 — synthetic SPR_BENCH (emb={emb_dim})"
            )
            fname = f"spr_bench_emb{emb_dim}_train_val_curves.png"
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating curve plot for emb_dim={emb_dim}: {e}")
            plt.close()

    # -------- aggregated bar chart of test Macro-F1 ----------
    try:
        dims, test_scores = [], []
        for emb_dim, logs in emb_logs.items():
            score = logs.get("test_macro_f1")
            if score is not None:
                dims.append(str(emb_dim))
                test_scores.append(score)

        if test_scores:
            plt.figure(figsize=(6, 4))
            plt.bar(dims, test_scores, color="skyblue")
            plt.xlabel("Embedding Dimension")
            plt.ylabel("Test Macro-F1")
            plt.title("Test Macro-F1 by Embedding Dim — synthetic SPR_BENCH")
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "spr_bench_test_f1_bar.png"))
            plt.close()
            print("Test macro-F1 scores:", dict(zip(dims, test_scores)))
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        plt.close()
