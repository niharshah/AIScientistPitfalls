import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------
# mandatory working dir
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------------------------------------------------------------
# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    runs = experiment_data.get("num_epochs", {})
except Exception as e:
    print(f"Error loading experiment data: {e}")
    runs = {}

# ---------------------------------------------------------------------
# 1) Per–run train/val BWA curves  (<=5 similar figures)
for i, (run_key, run_dict) in enumerate(runs.items()):
    if i >= 5:  # obey “at most 5 similar figures”
        break
    try:
        epochs = np.arange(1, len(run_dict["metrics"]["train"]) + 1)
        train_bwa = run_dict["metrics"]["train"]
        val_bwa = run_dict["metrics"]["val"]

        plt.figure()
        plt.plot(epochs, train_bwa, label="Train BWA")
        plt.plot(epochs, val_bwa, label="Validation BWA")
        plt.xlabel("Epoch")
        plt.ylabel("BWA")
        plt.title(f"SPR-BENCH BWA Learning Curve – {run_key}")
        plt.legend()
        plt.tight_layout()
        fname = f"spr_bench_bwa_curve_{run_key}.png"
        path = os.path.join(working_dir, fname)
        plt.savefig(path)
        plt.close()
        print(f"Saved {path}")
    except Exception as e:
        print(f"Error creating BWA curve for {run_key}: {e}")
        plt.close()

# ---------------------------------------------------------------------
# 2) Bar chart comparing test BWA across runs
try:
    run_names = []
    test_bwa_values = []
    for rk, rd in runs.items():
        if "test_metrics" in rd and "BWA" in rd["test_metrics"]:
            run_names.append(rk)
            test_bwa_values.append(rd["test_metrics"]["BWA"])

    plt.figure()
    x_pos = np.arange(len(run_names))
    plt.bar(x_pos, test_bwa_values, color="skyblue")
    plt.xticks(x_pos, run_names, rotation=45, ha="right")
    plt.ylabel("Test BWA")
    plt.title("SPR-BENCH: Test BWA for Different max_epoch Settings")
    plt.tight_layout()
    fname = "spr_bench_test_bwa_comparison.png"
    path = os.path.join(working_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")
except Exception as e:
    print(f"Error creating test BWA comparison bar chart: {e}")
    plt.close()

# ---------------------------------------------------------------------
# 3) Confusion matrix of best run (highest test BWA)
try:
    # locate best run
    best_run = max(
        runs.items(),
        key=lambda item: item[1].get("test_metrics", {}).get("BWA", -np.inf),
    )[0]
    preds = np.array(runs[best_run]["predictions"])
    gts = np.array(runs[best_run]["ground_truth"])
    num_classes = int(max(preds.max(), gts.max()) + 1)
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    for gt, pr in zip(gts, preds):
        conf_mat[gt, pr] += 1

    plt.figure(figsize=(6, 5))
    im = plt.imshow(conf_mat, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"SPR-BENCH Confusion Matrix – Best Run: {best_run}")
    for (i, j), v in np.ndenumerate(conf_mat):
        plt.text(j, i, str(v), ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    fname = f"spr_bench_confusion_matrix_{best_run}.png"
    path = os.path.join(working_dir, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
