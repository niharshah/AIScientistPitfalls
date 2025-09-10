import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- set up ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load ----------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_name = "SPR_BENCH"
ds_dict = experiment_data.get("hidden_dim", {}).get(ds_name, {})
hidden_dims = sorted(ds_dict.keys())


# helper to fetch arrays safely
def get_metric(hd, cat, field):
    return ds_dict.get(hd, {}).get(cat, {}).get(field, [])


# ---------- 1. loss curves ----------
try:
    plt.figure()
    for hd in hidden_dims:
        epochs = range(1, len(get_metric(hd, "losses", "train")) + 1)
        plt.plot(
            epochs,
            get_metric(hd, "losses", "train"),
            label=f"train h={hd}",
            linestyle="--",
        )
        plt.plot(epochs, get_metric(hd, "losses", "val"), label=f"val h={hd}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(
        "SPR_BENCH Training vs Validation Loss\nLeft: Train (dashed), Right: Val (solid)"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_loss_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------- 2. accuracy curves ----------
try:
    plt.figure()
    for hd in hidden_dims:
        epochs = range(1, len(get_metric(hd, "metrics", "train_acc")) + 1)
        plt.plot(
            epochs,
            get_metric(hd, "metrics", "train_acc"),
            label=f"train h={hd}",
            linestyle="--",
        )
        plt.plot(epochs, get_metric(hd, "metrics", "val_acc"), label=f"val h={hd}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "SPR_BENCH Training vs Validation Accuracy\nLeft: Train (dashed), Right: Val (solid)"
    )
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_accuracy_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# ---------- 3. URA curves ----------
try:
    plt.figure()
    for hd in hidden_dims:
        epochs = range(1, len(get_metric(hd, "metrics", "val_ura")) + 1)
        plt.plot(epochs, get_metric(hd, "metrics", "val_ura"), label=f"h={hd}")
    plt.xlabel("Epoch")
    plt.ylabel("Unseen Rule Accuracy (URA)")
    plt.title("SPR_BENCH Validation URA Across Hidden Sizes")
    plt.legend()
    fname = os.path.join(working_dir, "SPR_BENCH_val_URA_curves.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating URA plot: {e}")
    plt.close()

# ---------- 4. final test accuracy bar ----------
try:
    plt.figure()
    test_accs = [ds_dict[hd]["metrics"]["test_acc"] for hd in hidden_dims]
    plt.bar(range(len(hidden_dims)), test_accs, tick_label=hidden_dims)
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Test Accuracy")
    plt.title("SPR_BENCH Final Test Accuracy by Hidden Size")
    fname = os.path.join(working_dir, "SPR_BENCH_test_accuracy_bar.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test accuracy bar: {e}")
    plt.close()

# ---------- 5. final test URA bar ----------
try:
    plt.figure()
    test_uras = [ds_dict[hd]["metrics"]["test_ura"] for hd in hidden_dims]
    plt.bar(range(len(hidden_dims)), test_uras, tick_label=hidden_dims, color="orange")
    plt.xlabel("Hidden Dimension")
    plt.ylabel("Test URA")
    plt.title("SPR_BENCH Final Test URA by Hidden Size")
    fname = os.path.join(working_dir, "SPR_BENCH_test_URA_bar.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
    plt.close()
except Exception as e:
    print(f"Error creating test URA bar: {e}")
    plt.close()
