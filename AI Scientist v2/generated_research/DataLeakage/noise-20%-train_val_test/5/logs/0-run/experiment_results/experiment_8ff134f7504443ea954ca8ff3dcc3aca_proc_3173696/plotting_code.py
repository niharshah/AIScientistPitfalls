import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --------------- load experiment data ------------------ #
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}


# helper to fetch nested dict safely
def get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


results = get(
    experiment_data, "RemovePositionalEmbeddings", "SPR_BENCH", "results", default={}
)
if not results:
    print("No results found for SPR_BENCH. Exiting plotting script.")
else:
    # ----------- aggregate test accuracy plot ----------- #
    try:
        heads, test_accs = [], []
        for head, res in sorted(results.items(), key=lambda x: int(x[0])):
            acc = res.get("test_acc")
            if acc is not None:
                heads.append(int(head))
                test_accs.append(acc)
        if heads:
            plt.figure()
            plt.plot(heads, test_accs, marker="o")
            plt.xlabel("Number of Heads")
            plt.ylabel("Test Accuracy")
            plt.title(
                "SPR_BENCH: Test Accuracy vs Number of Attention Heads\n(RemovePositionalEmbeddings)"
            )
            fname = os.path.join(working_dir, "SPR_BENCH_accuracy_vs_heads.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy-vs-heads plot: {e}")
        plt.close()

    # ----------- per-head training curves (<=4) --------- #
    plotted = 0
    for head, res in sorted(results.items(), key=lambda x: int(x[0])):
        if plotted >= 4:  # at most 5 total plots including the aggregate above
            break
        try:
            metrics = res.get("metrics", {})
            losses = res.get("losses", {})
            epochs = range(1, len(metrics.get("train_acc", [])) + 1)
            if not epochs:
                continue
            plt.figure(figsize=(8, 4))
            # subplot 1: accuracy
            plt.subplot(1, 2, 1)
            plt.plot(epochs, metrics.get("train_acc", []), label="Train")
            plt.plot(epochs, metrics.get("val_acc", []), label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy")
            plt.legend()
            # subplot 2: loss
            plt.subplot(1, 2, 2)
            plt.plot(epochs, losses.get("train_loss", []), label="Train")
            plt.plot(epochs, losses.get("val_loss", []), label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss")
            plt.legend()
            plt.suptitle(
                f"SPR_BENCH Training Curves | nhead={head}\nLeft: Accuracy, Right: Loss"
            )
            fname = os.path.join(
                working_dir, f"SPR_BENCH_training_curves_nhead{head}.png"
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(fname)
            print(f"Saved {fname}")
            plotted += 1
            plt.close()
        except Exception as e:
            print(f"Error creating training curve for nhead={head}: {e}")
            plt.close()

    # -------------- print final metrics ----------------- #
    for head, res in sorted(results.items(), key=lambda x: int(x[0])):
        print(f"nhead={head}: test_acc={res.get('test_acc', 'N/A')}")
