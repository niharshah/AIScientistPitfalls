import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------------------------------
# basic set-up
# -------------------------------------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# -------------------------------------------------------
# 1. load every experiment_data.npy that the prompt lists
# -------------------------------------------------------
experiment_data_path_list = [
    "experiments/2025-08-16_02-31-48_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_dca8557aa7b645b5ac3278e5ddee5fdb_proc_3099932/experiment_data.npy",
    "experiments/2025-08-16_02-31-48_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_0c0758bb3d6144109759e7781358caba_proc_3099933/experiment_data.npy",
    "experiments/2025-08-16_02-31-48_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_27499ebabc834f069d08b8930fc901ff_proc_3099934/experiment_data.npy",
]

all_experiment_data = []
try:
    root = os.getenv("AI_SCIENTIST_ROOT", "")
    for exp_path in experiment_data_path_list:
        abs_path = os.path.join(root, exp_path)
        edata = np.load(abs_path, allow_pickle=True).item()
        all_experiment_data.append(edata)
    print(f"Loaded {len(all_experiment_data)} experiment_data dicts.")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# -------------------------------------------------------
# 2. aggregate per-dataset results
# -------------------------------------------------------
aggregated = {}  # dataset -> dict
for exp in all_experiment_data:
    if "num_epochs_sweep" not in exp:
        continue
    for dset, ddata in exp["num_epochs_sweep"].items():
        ag = aggregated.setdefault(
            dset,
            {
                "train_losses": [],  # list of np arrays
                "val_losses": [],
                "best_val": {},  # config_epochs -> list of metrics
            },
        )
        # store losses
        try:
            ag["train_losses"].append(np.asarray(ddata["losses"]["train"]))
            ag["val_losses"].append(np.asarray(ddata["losses"]["val"]))
        except Exception:
            pass
        # store best_val_metric per config_epochs
        try:
            for cfg_epochs, bval in zip(
                ddata["config_epochs"], ddata["best_val_metric"]
            ):
                ag["best_val"].setdefault(cfg_epochs, []).append(float(bval))
        except Exception:
            pass

# -------------------------------------------------------
# 3. plotting
# -------------------------------------------------------
for dset, ag in aggregated.items():
    # 3a. aggregated loss curves -------------------------------------------
    try:
        # align to shortest run so we can stack
        if ag["train_losses"] and ag["val_losses"]:
            min_len = min([len(a) for a in ag["train_losses"] + ag["val_losses"]])
            train_stack = np.stack([tl[:min_len] for tl in ag["train_losses"]])
            val_stack = np.stack([vl[:min_len] for vl in ag["val_losses"]])

            epochs = np.arange(1, min_len + 1)
            mean_train = train_stack.mean(axis=0)
            se_train = train_stack.std(axis=0, ddof=1) / np.sqrt(train_stack.shape[0])

            mean_val = val_stack.mean(axis=0)
            se_val = val_stack.std(axis=0, ddof=1) / np.sqrt(val_stack.shape[0])

            plt.figure()
            plt.plot(epochs, mean_train, label="Train – mean", color="tab:blue")
            plt.fill_between(
                epochs,
                mean_train - se_train,
                mean_train + se_train,
                color="tab:blue",
                alpha=0.3,
                label="Train ± SE",
            )
            plt.plot(epochs, mean_val, label="Val – mean", color="tab:orange")
            plt.fill_between(
                epochs,
                mean_val - se_val,
                mean_val + se_val,
                color="tab:orange",
                alpha=0.3,
                label="Val ± SE",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset} – Aggregated Training / Validation Loss (mean ± SE)")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset}_aggregated_loss_curves.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot for {dset}: {e}")
        plt.close()

    # 3b. aggregated best-val-metric vs max_epochs -------------------------
    try:
        if ag["best_val"]:
            cfg_vals = sorted(ag["best_val"].keys())
            means = [np.mean(ag["best_val"][c]) for c in cfg_vals]
            ses = [
                np.std(ag["best_val"][c], ddof=1) / np.sqrt(len(ag["best_val"][c]))
                for c in cfg_vals
            ]

            plt.figure()
            plt.bar(
                np.arange(len(cfg_vals)),
                means,
                yerr=ses,
                capsize=5,
                color="skyblue",
                alpha=0.8,
                label="Mean ± SE",
            )
            plt.xticks(np.arange(len(cfg_vals)), cfg_vals)
            plt.xlabel("max_epochs setting")
            plt.ylabel("Best Validation CWA-2D")
            plt.title(f"{dset} – Best Validation CWA-2D vs. max_epochs (mean ± SE)")
            plt.legend()
            fname = os.path.join(
                working_dir, f"{dset}_aggregated_CWA_vs_max_epochs.png"
            )
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated CWA plot for {dset}: {e}")
        plt.close()
