import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------------- load data safely -------------------------------------------
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

dataset_keys = list(experiment_data.keys())
colors = plt.cm.tab10.colors

# ---------------- 1) train / val macro-F1 curves -----------------------------
try:
    plt.figure()
    for i, k in enumerate(dataset_keys):
        ed = experiment_data[k]
        epochs = ed.get("epochs", [])
        tr = ed.get("metrics", {}).get("train_macro_f1", [])
        val = ed.get("metrics", {}).get("val_macro_f1", [])
        if not epochs:  # skip empty
            continue
        c = colors[i % len(colors)]
        plt.plot(epochs, tr, "--", color=c, label=f"{k}-train")
        plt.plot(epochs, val, "-", color=c, label=f"{k}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Macro-F1 Curves (Train dashed, Validation solid)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_macro_f1_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating macro-F1 plot: {e}")
    plt.close()

# ---------------- 2) train / val loss curves ---------------------------------
try:
    plt.figure()
    for i, k in enumerate(dataset_keys):
        ed = experiment_data[k]
        epochs = ed.get("epochs", [])
        tr = ed.get("losses", {}).get("train", [])
        val = ed.get("losses", {}).get("val", [])
        if not epochs:
            continue
        c = colors[i % len(colors)]
        plt.plot(epochs, tr, "--", color=c, label=f"{k}-train")
        plt.plot(epochs, val, "-", color=c, label=f"{k}-val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SPR_BENCH Loss Curves (Train dashed, Validation solid)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# ---------------- 3) pre-train loss curve ------------------------------------
try:
    plt.figure()
    for i, k in enumerate(dataset_keys):
        pre = experiment_data[k].get("losses", {}).get("pretrain", [])
        if not pre:
            continue
        c = colors[i % len(colors)]
        plt.plot(range(1, len(pre) + 1), pre, "-o", color=c, label=f"{k}")
    plt.xlabel("Pre-train Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Causal LM Pre-training Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "spr_bench_pretrain_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating pre-train loss plot: {e}")
    plt.close()

# ---------------- 4) test macro-F1 bar chart ---------------------------------
try:
    test_scores = {
        k: experiment_data[k].get("test_macro_f1", np.nan) for k in dataset_keys
    }
    plt.figure()
    plt.bar(
        range(len(test_scores)),
        list(test_scores.values()),
        tick_label=list(test_scores.keys()),
    )
    plt.ylabel("Macro-F1")
    plt.title("SPR_BENCH Test Macro-F1 per Setting")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "spr_bench_test_macro_f1_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating test bar chart: {e}")
    plt.close()

# ---------------- 5) confusion matrix heat-map --------------------------------
try:
    for k in dataset_keys:
        preds = np.array(experiment_data[k].get("predictions", []))
        gts = np.array(experiment_data[k].get("ground_truth", []))
        if preds.size == 0:
            continue
        n_cls = int(max(preds.max(), gts.max()) + 1)
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for p, t in zip(preds, gts):
            cm[t, p] += 1
        plt.figure()
        im = plt.imshow(cm, cmap="Blues")
        plt.colorbar(im)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"{k} Confusion Matrix (Test Set)")
        plt.savefig(os.path.join(working_dir, f"{k}_confusion_matrix.png"))
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# ---------------- numeric summary --------------------------------------------
summary = {k: experiment_data[k].get("test_macro_f1", np.nan) for k in dataset_keys}
print("Test Macro-F1 summary:", summary)
