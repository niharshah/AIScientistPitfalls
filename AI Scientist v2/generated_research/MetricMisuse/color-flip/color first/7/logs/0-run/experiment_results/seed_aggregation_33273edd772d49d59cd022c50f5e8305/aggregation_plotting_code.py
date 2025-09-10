import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load every experiment_data.npy that was listed by the user
experiment_data_path_list = [
    "experiments/2025-08-31_14-11-51_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_a8d888f085a848f3b005d703fb497747_proc_1743799/experiment_data.npy",
    "experiments/2025-08-31_14-11-51_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_7bcef6703ce24a41944b4335f9ea802b_proc_1743798/experiment_data.npy",
    "experiments/2025-08-31_14-11-51_symbol_glyph_clustering_attempt_0/logs/0-run/experiment_results/experiment_2e38fbaae2f444c5bac62db2aa0eb740_proc_1743796/experiment_data.npy",
]
all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_p = os.path.join(os.getenv("AI_SCIENTIST_ROOT", "."), p)
        if not os.path.isfile(full_p):
            print(f"Warning: file not found -> {full_p}")
            continue
        d = np.load(full_p, allow_pickle=True).item()
        all_experiment_data.append(d)
    if not all_experiment_data:
        raise RuntimeError("No experiment_data.npy files could be loaded.")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    raise SystemExit

# ------------------------------------------------------------------
# Aggregate metrics across runs
aggregated = {}  # {dataset_name: {metric_name: [runs x epochs ndarray] ...}}
for run_data in all_experiment_data:
    for dname, ed in run_data.items():
        if "epochs" not in ed or len(ed["epochs"]) == 0:
            continue
        epochs = np.array(ed["epochs"])
        losses = np.array(ed["losses"]["train"])
        train_metrics = ed["metrics"]["train"]
        val_metrics = ed["metrics"]["val"]

        # Extract metric arrays
        train_cpx = np.array([m["cpx"] for m in train_metrics])
        val_cpx = np.array([m["cpx"] for m in val_metrics])
        val_cwa = np.array([m["cwa"] for m in val_metrics])
        val_swa = np.array([m["swa"] for m in val_metrics])

        if dname not in aggregated:
            aggregated[dname] = {
                "epochs": epochs,
                "train_loss": [],
                "train_cpx": [],
                "val_cpx": [],
                "val_cwa": [],
                "val_swa": [],
            }

        # Make sure epochs align; if not, skip this run for that dataset
        if len(aggregated[dname]["epochs"]) != len(epochs) or not np.allclose(
            aggregated[dname]["epochs"], epochs
        ):
            print(f"Epoch mismatch for {dname} in one run; skipping that run.")
            continue

        aggregated[dname]["train_loss"].append(losses)
        aggregated[dname]["train_cpx"].append(train_cpx)
        aggregated[dname]["val_cpx"].append(val_cpx)
        aggregated[dname]["val_cwa"].append(val_cwa)
        aggregated[dname]["val_swa"].append(val_swa)

# Convert lists to numpy arrays for easier math
for dname, dct in aggregated.items():
    for k, v in dct.items():
        if k == "epochs":
            continue
        dct[k] = np.array(v)  # shape -> [runs, epochs]


# ------------------------------------------------------------------
def mean_sem(arr):
    """Return mean and standard error along axis 0."""
    mean = np.mean(arr, axis=0)
    sem = (
        np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


# ------------------------------------------------------------------
# Plot per-dataset aggregates
for dname, dct in aggregated.items():
    ep = dct["epochs"]

    # 1) Aggregated Training Loss
    try:
        if dct["train_loss"].size > 0:
            mean_loss, sem_loss = mean_sem(dct["train_loss"])
            plt.figure()
            plt.plot(ep, mean_loss, label="Mean Train Loss", color="tab:red")
            plt.fill_between(
                ep,
                mean_loss - sem_loss,
                mean_loss + sem_loss,
                alpha=0.3,
                color="tab:red",
                label="± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title(
                f"{dname}: Aggregated Training Loss per Epoch\nMean ± SEM across runs"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregated_train_loss.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregated train loss plot for {dname}: {e}")
        plt.close()

    # 2) Aggregated Train vs Val CpxWA
    try:
        if dct["train_cpx"].size > 0 and dct["val_cpx"].size > 0:
            m_tr, s_tr = mean_sem(dct["train_cpx"])
            m_val, s_val = mean_sem(dct["val_cpx"])
            plt.figure()
            plt.plot(ep, m_tr, label="Train CpxWA (mean)", color="tab:blue")
            plt.fill_between(
                ep,
                m_tr - s_tr,
                m_tr + s_tr,
                alpha=0.3,
                color="tab:blue",
                label="Train ± SEM",
            )
            plt.plot(ep, m_val, label="Val CpxWA (mean)", color="tab:green")
            plt.fill_between(
                ep,
                m_val - s_val,
                m_val + s_val,
                alpha=0.3,
                color="tab:green",
                label="Val ± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("CpxWA")
            plt.title(f"{dname}: Complexity-Weighted Accuracy\nMean ± SEM across runs")
            plt.legend()
            fname = os.path.join(working_dir, f"{dname}_aggregated_cpxwa_train_val.png")
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregated CpxWA plot for {dname}: {e}")
        plt.close()

    # 3) Aggregated Validation Weighted-Accuracy variants
    try:
        if (
            dct["val_cwa"].size > 0
            and dct["val_swa"].size > 0
            and dct["val_cpx"].size > 0
        ):
            m_cwa, s_cwa = mean_sem(dct["val_cwa"])
            m_swa, s_swa = mean_sem(dct["val_swa"])
            m_cpx, s_cpx = mean_sem(dct["val_cpx"])
            plt.figure()
            plt.plot(ep, m_cwa, marker="o", label="Val CWA (mean)")
            plt.fill_between(ep, m_cwa - s_cwa, m_cwa + s_cwa, alpha=0.3)
            plt.plot(ep, m_swa, marker="^", label="Val SWA (mean)")
            plt.fill_between(ep, m_swa - s_swa, m_swa + s_swa, alpha=0.3)
            plt.plot(ep, m_cpx, marker="s", label="Val CpxWA (mean)")
            plt.fill_between(ep, m_cpx - s_cpx, m_cpx + s_cpx, alpha=0.3)
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.title(
                f"{dname}: Validation Weighted Accuracies\nMean ± SEM across runs"
            )
            plt.legend()
            fname = os.path.join(
                working_dir, f"{dname}_aggregated_val_weighted_accuracy.png"
            )
            plt.savefig(fname)
            plt.close()
            print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating aggregated weighted accuracy plot for {dname}: {e}")
        plt.close()

# ------------------------------------------------------------------
# Cross-dataset comparison of aggregated validation CpxWA
if len(aggregated) > 1:
    try:
        plt.figure()
        for dname, dct in aggregated.items():
            if dct["val_cpx"].size == 0:
                continue
            m_val, s_val = mean_sem(dct["val_cpx"])
            ep = dct["epochs"]
            plt.plot(ep, m_val, label=f"{dname} (mean)")
            plt.fill_between(ep, m_val - s_val, m_val + s_val, alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("CpxWA")
        plt.title("Validation CpxWA Across Datasets\nMean ± SEM across runs")
        plt.legend()
        fname = os.path.join(working_dir, "cross_dataset_aggregated_val_cpxwa.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
    except Exception as e:
        print(f"Error creating cross-dataset aggregated plot: {e}")
        plt.close()
