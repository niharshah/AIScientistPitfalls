import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load data ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    exp = experiment_data.get("embedding_dim_tuning", {}).get("SPR_BENCH", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exp = {}

if exp:
    emb_dims = sorted(int(k.split("_")[-1]) for k in exp.keys())
    train_losses, val_losses, test_metrics = {}, {}, {}

    for ed in emb_dims:
        key = f"emb_dim_{ed}"
        d = exp[key]
        train_losses[ed] = d["losses"]["train"]
        val_losses[ed] = d["losses"]["val"]
        test_metrics[ed] = d["metrics"]["test"]

    # ---------- plot 1: training loss ----------
    try:
        plt.figure()
        for ed in emb_dims:
            plt.plot(
                range(1, len(train_losses[ed]) + 1),
                train_losses[ed],
                label=f"emb_dim={ed}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("SPR_BENCH Training Loss vs Epoch")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_training_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating training loss plot: {e}")
        plt.close()

    # ---------- plot 2: validation loss ----------
    try:
        plt.figure()
        for ed in emb_dims:
            plt.plot(
                range(1, len(val_losses[ed]) + 1), val_losses[ed], label=f"emb_dim={ed}"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("SPR_BENCH Validation Loss vs Epoch")
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_validation_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating validation loss plot: {e}")
        plt.close()

    # ---------- plot 3: test metrics ----------
    try:
        labels = ["CWA", "SWA", "GCWA"]
        x = np.arange(len(emb_dims))
        width = 0.25
        plt.figure()
        for i, m in enumerate(labels):
            vals = [test_metrics[ed][m] for ed in emb_dims]
            plt.bar(x + (i - 1) * width, vals, width, label=m)
        plt.xticks(x, [str(ed) for ed in emb_dims])
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(
            "SPR_BENCH Test Metrics across Embedding Dimensions\n"
            "Left: CWA, Center: SWA, Right: GCWA"
        )
        plt.legend()
        plt.savefig(os.path.join(working_dir, "SPR_BENCH_test_metrics.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating test metrics plot: {e}")
        plt.close()

    # ---------- print metrics ----------
    print("Test-set metrics by embedding dimension:")
    for ed in emb_dims:
        print(f"  emb_dim={ed}: {test_metrics[ed]}")
else:
    print("No experiment data found to plot.")
