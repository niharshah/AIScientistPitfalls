import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- basic set-up ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load every run ----------
experiment_data_path_list = [
    "experiments/2025-08-17_23-44-17_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_2a759f81326d4a2381cc66d354015197_proc_3471779/experiment_data.npy",
    "experiments/2025-08-17_23-44-17_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_83ca157fb0d74585b17303828577c6ab_proc_3471777/experiment_data.npy",
    "experiments/2025-08-17_23-44-17_conceptual_generalization_poly_rule_attempt_0/logs/0-run/experiment_results/experiment_2aa8a7297a8640949d9fb806ec5016e4_proc_3471780/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {p}: {e}")

# ---------- aggregate by dataset ----------
agg = {}
for run in all_experiment_data:
    for dset, rec in run.items():
        ds = agg.setdefault(
            dset,
            {
                "epochs": [],
                "train_f1": [],
                "val_f1": [],
                "train_loss": [],
                "val_loss": [],
                "test_f1": [],
            },
        )
        epochs = np.asarray(rec.get("epochs", []))
        m = rec.get("metrics", {})
        l = rec.get("losses", {})
        if len(epochs):
            ds["epochs"].append(epochs)
        if m.get("train_f1") is not None:
            ds["train_f1"].append(np.asarray(m["train_f1"]))
        if m.get("val_f1") is not None:
            ds["val_f1"].append(np.asarray(m["val_f1"]))
        if l.get("train") is not None:
            ds["train_loss"].append(np.asarray(l["train"]))
        if l.get("val") is not None:
            ds["val_loss"].append(np.asarray(l["val"]))
        test_f1 = m.get("test_f1")
        if test_f1 is not None:
            ds["test_f1"].append(test_f1)


# ---------- helper ----------
def stack_and_crop(list_of_arr):
    """stack 1D arrays along axis 0, cropping to min length"""
    if len(list_of_arr) == 0:
        return None
    min_len = min([len(a) for a in list_of_arr])
    arr = np.stack([a[:min_len] for a in list_of_arr], axis=0)
    return arr


# ---------- create plots ----------
for dset, rec in agg.items():
    # epochs (use first run, already cropped)
    ep_stack = stack_and_crop(rec["epochs"])
    if ep_stack is None:
        continue
    epochs = ep_stack[0]  # all equal length after crop

    # ----- F1 curves -----
    try:
        tr_stack = stack_and_crop(rec["train_f1"])
        val_stack = stack_and_crop(rec["val_f1"])
        if tr_stack is not None and val_stack is not None:
            tr_mean, tr_sem = tr_stack.mean(0), tr_stack.std(0) / np.sqrt(
                tr_stack.shape[0]
            )
            val_mean, val_sem = val_stack.mean(0), val_stack.std(0) / np.sqrt(
                val_stack.shape[0]
            )
            plt.figure()
            plt.plot(epochs, tr_mean, label="Train mean")
            plt.fill_between(
                epochs,
                tr_mean - tr_sem,
                tr_mean + tr_sem,
                alpha=0.3,
                label="Train ±SEM",
            )
            plt.plot(epochs, val_mean, linestyle="--", label="Validation mean")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                alpha=0.3,
                label="Val ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Macro-F1")
            plt.title(
                f"{dset}: Aggregated Train/Val Macro-F1\n(shaded: ±SEM, n={tr_stack.shape[0]})"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_aggregate_f1_curves.png"))
    except Exception as e:
        print(f"Error creating aggregated F1 for {dset}: {e}")
    finally:
        plt.close()

    # ----- Loss curves -----
    try:
        tr_l_stack = stack_and_crop(rec["train_loss"])
        val_l_stack = stack_and_crop(rec["val_loss"])
        if tr_l_stack is not None and val_l_stack is not None:
            tr_mean, tr_sem = tr_l_stack.mean(0), tr_l_stack.std(0) / np.sqrt(
                tr_l_stack.shape[0]
            )
            val_mean, val_sem = val_l_stack.mean(0), val_l_stack.std(0) / np.sqrt(
                val_l_stack.shape[0]
            )
            plt.figure()
            plt.plot(epochs, tr_mean, label="Train mean")
            plt.fill_between(
                epochs,
                tr_mean - tr_sem,
                tr_mean + tr_sem,
                alpha=0.3,
                label="Train ±SEM",
            )
            plt.plot(epochs, val_mean, linestyle="--", label="Validation mean")
            plt.fill_between(
                epochs,
                val_mean - val_sem,
                val_mean + val_sem,
                alpha=0.3,
                label="Val ±SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{dset}: Aggregated Train/Val Loss\n(shaded: ±SEM, n={tr_l_stack.shape[0]})"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{dset}_aggregate_loss_curves.png"))
    except Exception as e:
        print(f"Error creating aggregated Loss for {dset}: {e}")
    finally:
        plt.close()

# ---------- aggregated bar chart of test-F1 ----------
try:
    names, means, sems = [], [], []
    for dset, rec in agg.items():
        if len(rec["test_f1"]):
            arr = np.asarray(rec["test_f1"], dtype=float)
            names.append(dset)
            means.append(arr.mean())
            sems.append(arr.std() / np.sqrt(len(arr)))
    if names:
        idx = np.arange(len(names))
        plt.figure()
        plt.bar(idx, means, yerr=sems, capsize=5, color="skyblue")
        plt.xticks(idx, names, rotation=45, ha="right")
        for i, (m, s) in enumerate(zip(means, sems)):
            plt.text(i, m + 0.01, f"{m:.3f}±{s:.3f}", ha="center")
        plt.ylabel("Macro-F1")
        plt.title(
            "Aggregated Test Macro-F1 Across Datasets (bars: mean, whiskers: ±SEM)"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "aggregate_test_f1_comparison.png"))
except Exception as e:
    print(f"Error creating aggregated test F1 bar chart: {e}")
finally:
    plt.close()

# ---------- console summary ----------
print("=== Aggregated Test Macro-F1 (mean ± SEM) ===")
for n, m, s in zip(names, means, sems):
    print(f"{n}: {m:.4f} ± {s:.4f}")
