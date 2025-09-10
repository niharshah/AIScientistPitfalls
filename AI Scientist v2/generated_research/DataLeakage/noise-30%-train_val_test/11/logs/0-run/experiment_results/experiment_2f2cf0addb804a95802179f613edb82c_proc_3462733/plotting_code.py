import matplotlib.pyplot as plt
import numpy as np
import os

# set up working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch data safely
def get_spr_logs():
    try:
        return experiment_data["emb_dim_tuning"]["SPR_BENCH"]
    except Exception as e:
        print(f"Cannot find SPR_BENCH logs: {e}")
        return {}


spr_logs = get_spr_logs()
embed_keys = sorted(
    spr_logs.keys(), key=lambda x: int(x.split("_")[-1])
)  # ['emb_128', ...]

# 1) Loss curves (train & val) for all emb dims
try:
    plt.figure(figsize=(7, 5))
    for k in embed_keys:
        epochs = spr_logs[k]["epochs"]
        plt.plot(
            epochs, spr_logs[k]["losses"]["train"], label=f"{k}-train", linestyle="-"
        )
        plt.plot(epochs, spr_logs[k]["losses"]["val"], label=f"{k}-val", linestyle="--")
    plt.title(
        "SPR_BENCH: Training vs Validation Loss\n(Left: Train, Right: Val curves by embedding dim)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2) Macro-F1 curves (train & val) for all emb dims
try:
    plt.figure(figsize=(7, 5))
    for k in embed_keys:
        epochs = spr_logs[k]["epochs"]
        plt.plot(
            epochs,
            spr_logs[k]["metrics"]["train_macro_f1"],
            label=f"{k}-train",
            linestyle="-",
        )
        plt.plot(
            epochs,
            spr_logs[k]["metrics"]["val_macro_f1"],
            label=f"{k}-val",
            linestyle="--",
        )
    plt.title(
        "SPR_BENCH: Training vs Validation Macro-F1\n(Left: Train, Right: Val curves by embedding dim)"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend(fontsize=7)
    fname = os.path.join(working_dir, "SPR_BENCH_macro_f1_curves.png")
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating F1 curve plot: {e}")
    plt.close()

# 3) Test Macro-F1 by embedding dimension
try:
    plt.figure(figsize=(6, 4))
    dims = [int(k.split("_")[-1]) for k in embed_keys]
    test_f1 = [spr_logs[k]["test_macro_f1"] for k in embed_keys]
    plt.bar(range(len(dims)), test_f1, tick_label=dims)
    plt.title("SPR_BENCH: Test Macro-F1 vs Embedding Dim\nBar height = Test Macro-F1")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Test Macro-F1")
    fname = os.path.join(working_dir, "SPR_BENCH_test_macro_f1_bar.png")
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test F1 bar plot: {e}")
    plt.close()
