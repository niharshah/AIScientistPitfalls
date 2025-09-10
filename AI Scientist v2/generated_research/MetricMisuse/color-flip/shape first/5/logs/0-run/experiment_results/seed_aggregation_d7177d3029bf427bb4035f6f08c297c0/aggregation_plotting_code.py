import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

# ------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------
# Discover all experiment_data.npy files that live anywhere below cwd
# or respect an optional colon–separated list in the env var EXP_DATA
exp_paths_env = os.getenv("EXP_DATA", "")
if exp_paths_env.strip():
    experiment_data_path_list = [p for p in exp_paths_env.split(":") if p.strip()]
else:
    experiment_data_path_list = glob("**/experiment_data.npy", recursive=True)

if not experiment_data_path_list:
    print("No experiment_data.npy files were found.")
    experiment_data_path_list = []

# ------------------------------------------------------------
# Load all files
all_experiment_data = []
for path in experiment_data_path_list:
    try:
        data = np.load(path, allow_pickle=True).item()
        all_experiment_data.append(data)
    except Exception as e:
        print(f"Error loading {path}: {e}")

# ------------------------------------------------------------
# Group runs by dataset key
dataset_runs = {}  # {dataset_key: [run_dict, run_dict, ...]}
for exp in all_experiment_data:
    for dset_key, dset_val in exp.items():
        dataset_runs.setdefault(dset_key, []).append(dset_val)


# ------------------------------------------------------------
# Helper for standard error
def stderr(arr, axis=0):
    return np.std(arr, axis=axis, ddof=1) / np.sqrt(arr.shape[axis])


# ------------------------------------------------------------
for dset_key, runs in dataset_runs.items():
    # Skip datasets that have no metrics
    if not runs:
        continue

    # --------- Prepare per-epoch arrays (train, val) ----------
    # Align runs by shortest common length
    try:
        train_arrays, val_arrays, loss_train_arrays, loss_val_arrays, test_vals = (
            [],
            [],
            [],
            [],
            [],
        )
        min_len = None
        for r in runs:
            try:
                tr = np.asarray(r["metrics"]["train"])
                vl = np.asarray(r["metrics"]["val"])
                lt = np.asarray(r["losses"]["train"])
                lv = np.asarray(r["losses"]["val"])
                cur_len = min(len(tr), len(vl), len(lt), len(lv))
                if min_len is None or cur_len < min_len:
                    min_len = cur_len
            except Exception:
                continue  # skip malformed run

        if min_len is None or min_len == 0:
            print(f"{dset_key}: could not determine epoch length; skipping plots.")
            continue

        # Collect aligned arrays
        for r in runs:
            try:
                tr = np.asarray(r["metrics"]["train"])[:min_len]
                vl = np.asarray(r["metrics"]["val"])[:min_len]
                lt = np.asarray(r["losses"]["train"])[:min_len]
                lv = np.asarray(r["losses"]["val"])[:min_len]
                train_arrays.append(tr)
                val_arrays.append(vl)
                loss_train_arrays.append(lt)
                loss_val_arrays.append(lv)
                if r["metrics"]["test"]:
                    test_vals.append(float(r["metrics"]["test"][0]))
            except Exception:
                continue

        # Convert to np.array for vector ops
        train_arrays = np.vstack(train_arrays)
        val_arrays = np.vstack(val_arrays)
        loss_train_arrays = np.vstack(loss_train_arrays)
        loss_val_arrays = np.vstack(loss_val_arrays)
        epochs_axis = np.arange(1, min_len + 1)

        # --------- Plot 1: CWCA mean ± SE curves ----------
        try:
            plt.figure()
            mean_train = train_arrays.mean(axis=0)
            se_train = stderr(train_arrays, axis=0)
            mean_val = val_arrays.mean(axis=0)
            se_val = stderr(val_arrays, axis=0)

            plt.plot(epochs_axis, mean_train, label="Train Mean", color="steelblue")
            plt.fill_between(
                epochs_axis,
                mean_train - se_train,
                mean_train + se_train,
                color="steelblue",
                alpha=0.3,
                label="Train ± SE",
            )

            plt.plot(
                epochs_axis, mean_val, label="Val Mean", color="orange", linestyle="--"
            )
            plt.fill_between(
                epochs_axis,
                mean_val - se_val,
                mean_val + se_val,
                color="orange",
                alpha=0.3,
                label="Val ± SE",
            )

            plt.xlabel("Epoch")
            plt.ylabel("CWCA")
            plt.title(f"{dset_key} – Aggregated CWCA Curves\nMean ± Standard Error")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_key}_cwca_mean_se.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated CWCA plot for {dset_key}: {e}")
            plt.close()

        # --------- Plot 2: Loss mean ± SE curves ----------
        try:
            plt.figure()
            mean_lt = loss_train_arrays.mean(axis=0)
            se_lt = stderr(loss_train_arrays, axis=0)
            mean_lv = loss_val_arrays.mean(axis=0)
            se_lv = stderr(loss_val_arrays, axis=0)

            plt.plot(epochs_axis, mean_lt, label="Train Loss Mean", color="green")
            plt.fill_between(
                epochs_axis,
                mean_lt - se_lt,
                mean_lt + se_lt,
                color="green",
                alpha=0.3,
                label="Train ± SE",
            )

            plt.plot(
                epochs_axis, mean_lv, label="Val Loss Mean", color="red", linestyle="--"
            )
            plt.fill_between(
                epochs_axis,
                mean_lv - se_lv,
                mean_lv + se_lv,
                color="red",
                alpha=0.3,
                label="Val ± SE",
            )

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{dset_key} – Aggregated Loss Curves\nMean ± Standard Error")
            plt.legend()
            fname = os.path.join(working_dir, f"{dset_key}_loss_mean_se.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss plot for {dset_key}: {e}")
            plt.close()

        # --------- Plot 3: Test CWCA bar with error ----------
        try:
            if test_vals:
                test_vals = np.asarray(test_vals)
                mean_test = test_vals.mean()
                se_test = test_vals.std(ddof=1) / np.sqrt(len(test_vals))

                plt.figure()
                plt.bar(
                    [0],
                    [mean_test],
                    yerr=[se_test],
                    color="purple",
                    capsize=8,
                    label="Mean ± SE",
                )
                plt.ylim(0, 1.0)
                plt.xticks([0], ["Test"])
                plt.ylabel("CWCA")
                plt.title(f"{dset_key} – Final Test CWCA\nMean ± Standard Error")
                plt.text(
                    0,
                    mean_test + 0.03,
                    f"{mean_test:.3f}±{se_test:.3f}",
                    ha="center",
                    va="bottom",
                )
                plt.legend()
                fname = os.path.join(working_dir, f"{dset_key}_test_cwca_mean_se.png")
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
            else:
                print(f"{dset_key}: No test CWCA values available.")
        except Exception as e:
            print(f"Error creating aggregated test CWCA plot for {dset_key}: {e}")
            plt.close()

        # --------- Print final aggregated metric ----------
        if test_vals.size:
            print(
                f"{dset_key} – Final Test CWCA: {mean_test:.4f} ± {se_test:.4f} (n={len(test_vals)})"
            )
    except Exception as e:
        print(f"Error processing dataset {dset_key}: {e}")
