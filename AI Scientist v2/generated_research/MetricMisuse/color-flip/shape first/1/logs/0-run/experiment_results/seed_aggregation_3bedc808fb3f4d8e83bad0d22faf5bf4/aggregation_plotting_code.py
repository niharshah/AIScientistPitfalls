import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- basic setup ----------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ---------- load all experiment_data.npy files ----------
experiment_data_path_list = [
    "experiments/2025-08-15_22-24-43_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_dec6e9079e9f493a9aa826d4379752bb_proc_2997849/experiment_data.npy",
    "experiments/2025-08-15_22-24-43_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_8e6554cf551945fe8e4b7d0a11871810_proc_2997847/experiment_data.npy",
    "experiments/2025-08-15_22-24-43_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_707f2681341f44389f0c8f9cfef7a073_proc_2997848/experiment_data.npy",
]

all_experiment_data = []
for p in experiment_data_path_list:
    try:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
    except Exception as e:
        print(f"Error loading {p}: {e}")


# ---------- aggregation helpers ----------
def pad_to_same_length(arr_list, pad_val=np.nan):
    max_len = max(len(a) for a in arr_list)
    padded = []
    for a in arr_list:
        pad_size = max_len - len(a)
        if pad_size:
            padded.append(np.concatenate([a, np.full(pad_size, pad_val)]))
        else:
            padded.append(np.asarray(a))
    return np.vstack(padded)


def mean_se(arr_2d):
    mean = np.nanmean(arr_2d, axis=0)
    se = np.nanstd(arr_2d, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr_2d), axis=0))
    return mean, se


# aggregated[dset][pt_k]['losses'|'metrics'][name] = 2-tuple (mean, se)
aggregated = {}

for ed in all_experiment_data:
    for dset, runs_dict in ed.get("pretrain_epochs", {}).items():
        ds_agg = aggregated.setdefault(dset, {})
        for k, run in runs_dict.items():
            k_agg = ds_agg.setdefault(k, {"losses": {}, "metrics": {}})
            # ------ losses ------
            for split, losses in run.get(
                "losses", {}
            ).items():  # pretrain / train / val
                k_agg["losses"].setdefault(split, []).append(
                    np.asarray(losses, dtype=float)
                )
            # ------ metrics ------
            for m_name, vals in run.get("metrics", {}).items():
                k_agg["metrics"].setdefault(m_name, []).append(
                    np.asarray(vals, dtype=float)
                )

# convert lists to (mean,se)
for dset in aggregated:
    for k in aggregated[dset]:
        # losses
        for split, lst in aggregated[dset][k]["losses"].items():
            arr = pad_to_same_length(lst)
            aggregated[dset][k]["losses"][split] = mean_se(arr)
        # metrics
        for m_name, lst in aggregated[dset][k]["metrics"].items():
            arr = pad_to_same_length(lst)
            aggregated[dset][k]["metrics"][m_name] = mean_se(arr)


# --------- plotting style helper ----------
def _style(idx):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[idx % len(colors)]


# --------- generate plots ----------
for dset, ds_dict in aggregated.items():
    run_keys = sorted(ds_dict.keys(), key=lambda x: int(x))  # e.g. ['2','4',...]

    # 1. aggregated pre-training loss
    try:
        plt.figure()
        for i, k in enumerate(run_keys):
            mean, se = ds_dict[k]["losses"].get("pretrain", (None, None))
            if mean is None:
                continue
            epochs = np.arange(1, len(mean) + 1)
            c = _style(i)
            plt.plot(epochs, mean, color=c, label=f"Mean PT={k}")
            plt.fill_between(
                epochs, mean - se, mean + se, color=c, alpha=0.25, label=f"SE PT={k}"
            )
        plt.title(f"{dset}: Pre-training Loss (Mean ± SE)")
        plt.xlabel("Pre-training Epoch")
        plt.ylabel("Loss")
        plt.legend(fontsize="small", ncol=2)
        fname = os.path.join(working_dir, f"{dset}_pretrain_loss_agg.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated pre-training loss plot: {e}")
        plt.close()

    # 2. aggregated fine-tuning loss (train/val)
    try:
        plt.figure()
        ls_map = {"train": "-", "val": "--"}
        for i, k in enumerate(run_keys):
            for split in ["train", "val"]:
                if split not in ds_dict[k]["losses"]:
                    continue
                mean, se = ds_dict[k]["losses"][split]
                epochs = np.arange(1, len(mean) + 1)
                c = _style(i)
                plt.plot(
                    epochs,
                    mean,
                    color=c,
                    linestyle=ls_map[split],
                    label=f"{split.capitalize()} PT={k}",
                )
                plt.fill_between(epochs, mean - se, mean + se, color=c, alpha=0.25)
        plt.title(f"{dset}: Fine-tuning Loss (Mean ± SE)")
        plt.xlabel("Fine-tuning Epoch")
        plt.ylabel("Loss")
        plt.legend(fontsize="small", ncol=2)
        fname = os.path.join(working_dir, f"{dset}_finetune_loss_agg.png")
        plt.savefig(fname)
        print(f"Saved {fname}")
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated fine-tune loss plot: {e}")
        plt.close()

    # 3+. aggregated metrics
    metric_names = set()
    for k in run_keys:
        metric_names.update(ds_dict[k]["metrics"].keys())

    for m_name in metric_names:
        try:
            plt.figure()
            for i, k in enumerate(run_keys):
                if m_name not in ds_dict[k]["metrics"]:
                    continue
                mean, se = ds_dict[k]["metrics"][m_name]
                epochs = np.arange(1, len(mean) + 1)
                c = _style(i)
                plt.plot(epochs, mean, color=c, label=f"Mean PT={k}")
                plt.fill_between(
                    epochs,
                    mean - se,
                    mean + se,
                    color=c,
                    alpha=0.25,
                    label=f"SE PT={k}" if i == 0 else None,
                )
                # print last epoch stats
                final_mean, final_se = mean[-1], se[-1]
                print(
                    f"{dset} | {m_name} | PT={k} | final mean±SE: {final_mean:.4f} ± {final_se:.4f}"
                )
            plt.title(f"{dset}: {m_name} (Mean ± SE) across Fine-tuning Epochs")
            plt.xlabel("Fine-tuning Epoch")
            plt.ylabel(m_name)
            plt.legend(fontsize="small")
            fname = os.path.join(working_dir, f"{dset}_{m_name}_agg.png")
            plt.savefig(fname)
            print(f"Saved {fname}")
            plt.close()
        except Exception as e:
            print(f"Error creating aggregated {m_name} plot: {e}")
            plt.close()
