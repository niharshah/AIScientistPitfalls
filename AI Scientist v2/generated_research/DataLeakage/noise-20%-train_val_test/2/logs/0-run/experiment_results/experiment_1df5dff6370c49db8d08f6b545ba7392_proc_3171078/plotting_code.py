import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- paths ----------
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


# ---------- helper ----------
def cmatrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ---------- plot 1: pretraining loss ----------
try:
    ed = experiment_data.get("pretrain+cls", {})
    plt.figure()
    plt.plot(
        range(1, len(ed.get("losses", {}).get("pretrain", [])) + 1),
        ed.get("losses", {}).get("pretrain", []),
        marker="o",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pretraining LM Loss (SPR Bench)")
    plt.tight_layout()
    fname = os.path.join(working_dir, "sprbench_pretrain_loss.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating pretraining loss plot: {e}")
    plt.close()

# ---------- plot 2: train / val loss ----------
try:
    plt.figure()
    for tag, color in zip(["pretrain+cls", "scratch_cls"], ["tab:blue", "tab:orange"]):
        ed = experiment_data[tag]
        ep = ed["epochs"]
        plt.plot(
            ep, ed["losses"]["train"], label=f"{tag} train", linestyle="--", color=color
        )
        plt.plot(
            ep, ed["losses"]["val"], label=f"{tag} val", linestyle="-", color=color
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss (SPR Bench)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "sprbench_train_val_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating train/val loss plot: {e}")
    plt.close()

# ---------- plot 3: train / val macro-F1 ----------
try:
    plt.figure()
    for tag, color in zip(["pretrain+cls", "scratch_cls"], ["tab:green", "tab:red"]):
        ed = experiment_data[tag]
        ep = ed["epochs"]
        plt.plot(
            ep,
            ed["metrics"]["train_macro_f1"],
            label=f"{tag} train",
            linestyle="--",
            color=color,
        )
        plt.plot(
            ep,
            ed["metrics"]["val_macro_f1"],
            label=f"{tag} val",
            linestyle="-",
            color=color,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Train / Validation Macro-F1 (SPR Bench)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "sprbench_train_val_macro_f1.png"))
    plt.close()
except Exception as e:
    print(f"Error creating train/val macro-F1 plot: {e}")
    plt.close()

# ---------- plot 4: val macro-F1 comparison ----------
try:
    plt.figure()
    for tag, style in zip(["pretrain+cls", "scratch_cls"], ["-o", "-s"]):
        ed = experiment_data[tag]
        plt.plot(ed["epochs"], ed["metrics"]["val_macro_f1"], style, label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("Val Macro-F1")
    plt.title("Validation Macro-F1 Comparison (SPR Bench)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "sprbench_val_macro_f1_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val macro-F1 comparison plot: {e}")
    plt.close()

# ---------- plot 5: confusion matrix ----------
try:
    # pick best experiment by last val macro-F1
    best_tag = max(
        ["pretrain+cls", "scratch_cls"],
        key=lambda t: experiment_data[t]["metrics"]["val_macro_f1"][-1],
    )
    ed = experiment_data[best_tag]
    y_true = np.array(ed["ground_truth"])
    y_pred = np.array(ed["predictions"])
    num_classes = len(np.unique(y_true))
    cm = cmatrix(y_true, y_pred, num_classes)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (SPR Bench) – {best_tag}")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"sprbench_confmat_{best_tag}.png"))
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
