import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

root = experiment_data.get("MultiSyntheticDatasets", {})
for task_name, task_data in root.items():
    # Gather nhead keys sorted as int
    nheads = sorted(task_data.keys(), key=lambda x: int(x))
    # ---------- Accuracy Curves ----------
    try:
        plt.figure()
        for nh in nheads:
            log = task_data[nh]
            epochs = np.arange(1, len(log["metrics"]["train"]) + 1)
            plt.plot(
                epochs, log["metrics"]["train"], label=f"nhead={nh} train", marker="o"
            )
            plt.plot(
                epochs,
                log["metrics"]["val"],
                label=f"nhead={nh} val",
                linestyle="--",
                marker="x",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"MultiSyntheticDatasets – {task_name} – Accuracy Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"{task_name}_accuracy_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {task_name}: {e}")
        plt.close()

    # ---------- Loss Curves ----------
    try:
        plt.figure()
        for nh in nheads:
            log = task_data[nh]
            epochs = np.arange(1, len(log["losses"]["train"]) + 1)
            plt.plot(
                epochs, log["losses"]["train"], label=f"nhead={nh} train", marker="o"
            )
            plt.plot(
                epochs,
                log["losses"]["val"],
                label=f"nhead={nh} val",
                linestyle="--",
                marker="x",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.title(f"MultiSyntheticDatasets – {task_name} – Loss Curves")
        plt.legend()
        fname = os.path.join(working_dir, f"{task_name}_loss_curves.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {task_name}: {e}")
        plt.close()

    # ---------- Test Accuracy Bar ----------
    try:
        plt.figure()
        test_accs = [task_data[nh]["test_acc"] for nh in nheads]
        plt.bar([str(nh) for nh in nheads], test_accs, color="skyblue")
        plt.ylim(0, 1)
        plt.xlabel("nhead")
        plt.ylabel("Test Accuracy")
        plt.title(f"MultiSyntheticDatasets – {task_name} – Test Accuracy")
        for idx, acc in enumerate(test_accs):
            plt.text(idx, acc + 0.01, f"{acc:.2f}", ha="center")
        fname = os.path.join(working_dir, f"{task_name}_test_accuracy.png")
        plt.savefig(fname)
        print("Saved", fname)
        plt.close()
    except Exception as e:
        print(f"Error creating test-accuracy plot for {task_name}: {e}")
        plt.close()
