import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data and "SPR_BENCH" in experiment_data:
    log = experiment_data["SPR_BENCH"]

    # ---------- 1) learning curves ----------
    try:
        epochs = range(1, len(log["losses"]["train"]) + 1)
        tr, vl = log["losses"]["train"], log["losses"]["val"]
        swa = [m.get("swa", np.nan) for m in log["metrics"]["val"]]

        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(epochs, tr, "b-o", label="Train Loss")
        ax1.plot(epochs, vl, "r-o", label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(epochs, swa, "g-s", label="Val SWA")
        ax2.set_ylabel("SWA")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")

        plt.title("SPR_BENCH Learning Curves\nLeft: Loss, Right: SWA")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_learning_curves.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating learning curve plot: {e}")
        plt.close()

    # ---------- 2) test SWA bar ----------
    try:
        test_swa = log["metrics"]["test"]["swa"]
        plt.figure(figsize=(3, 4))
        plt.bar(["Test"], [test_swa], color="steelblue")
        plt.ylim(0, 1)
        plt.title("SPR_BENCH: Test SWA")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_test_swa_bar.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating test SWA bar: {e}")
        plt.close()

    # ---------- 3) confusion matrix ----------
    try:
        y_t, y_p = np.array(log["ground_truth"]), np.array(log["predictions"])
        labels = np.unique(np.concatenate([y_t, y_p]))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for t, p in zip(y_t, y_p):
            cm[t, p] += 1
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("SPR_BENCH Confusion Matrix")
        plt.xticks(labels)
        plt.yticks(labels)
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_confusion_matrix.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        plt.close()

    # ---------- 4) per-class accuracy ----------
    try:
        accs = [(y_p[y_t == l] == l).mean() if (y_t == l).any() else 0 for l in labels]
        plt.figure(figsize=(6, 4))
        plt.bar([str(l) for l in labels], accs, color="orange")
        plt.ylim(0, 1)
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.title("SPR_BENCH Per-Class Accuracy")
        plt.tight_layout()
        fname = os.path.join(working_dir, "SPR_BENCH_per_class_accuracy.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating per-class accuracy plot: {e}")
        plt.close()
else:
    print("No valid experiment data found.")
