import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ------------------------------------------------------------------
# Load every experiment_data.npy listed by the user
# ------------------------------------------------------------------
try:
    experiment_data_path_list = [
        "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_fc93043d380a440d9c8603d75a178d9d_proc_3087392/experiment_data.npy",
        "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_a6016aa3191a4a5cbfc8ae0a15bf9af1_proc_3087394/experiment_data.npy",
        "experiments/2025-08-16_00-47-34_context_aware_contrastive_learning_attempt_0/logs/0-run/experiment_results/experiment_af317d6e91104d2d977f5e7a61b3fa12_proc_3087391/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        abs_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        ed = np.load(abs_path, allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# Helper -------------------------------------------------------------------------------------------
def unpack_pairs(pair_list):
    if not pair_list:
        return []
    return [(ep, val) for ep, val in pair_list]


def build_epoch_dict(list_of_pairs):
    """Return dict epoch -> value"""
    return {ep: v for ep, v in list_of_pairs}


def aggregate_across_runs(run_dicts):
    """
    run_dicts: list[dict] where each maps epoch->scalar
    Returns sorted_epochs, mean_vals, sem_vals
    """
    if not run_dicts:
        return [], [], []
    epochs = sorted(set().union(*[d.keys() for d in run_dicts]))
    mean_vals, sem_vals = [], []
    for ep in epochs:
        vals = [d[ep] for d in run_dicts if ep in d]
        if not vals:
            mean_vals.append(np.nan)
            sem_vals.append(np.nan)
        else:
            vals = np.array(vals, dtype=float)
            mean_vals.append(vals.mean())
            sem_vals.append(vals.std(ddof=1) / np.sqrt(len(vals)))
    return epochs, mean_vals, sem_vals


# Extract all SPR blocks ---------------------------------------------------------------------------
spr_runs = []
for ed in all_experiment_data:
    try:
        spr_runs.append(ed["last_token_repr"]["SPR"])
    except Exception:
        pass  # ignore runs where SPR is missing

# Nothing to plot if empty
if not spr_runs:
    print("No SPR data found in any experiment files.")
else:
    # 1. Contrastive pre-training loss --------------------------------------------------------------
    try:
        run_dicts = []
        for spr in spr_runs:
            pairs = unpack_pairs(spr["contrastive_pretrain"]["losses"])
            run_dicts.append(build_epoch_dict(pairs))
        ep, mean_v, sem_v = aggregate_across_runs(run_dicts)

        if ep:
            plt.figure()
            plt.plot(ep, mean_v, label="Mean loss", color="tab:blue")
            plt.fill_between(
                ep,
                np.array(mean_v) - np.array(sem_v),
                np.array(mean_v) + np.array(sem_v),
                color="tab:blue",
                alpha=0.3,
                label="± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("NT-Xent Loss")
            plt.title("SPR Contrastive Pre-training Loss (Aggregated)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "SPR_contrastive_loss_agg.png"))
            plt.close()
            print("Aggregated contrastive loss curve saved.")
    except Exception as e:
        print(f"Error creating aggregated contrastive loss plot: {e}")
        plt.close()

    # 2. Fine-tune train & val loss -----------------------------------------------------------------
    try:
        run_dicts_tr, run_dicts_val = [], []
        for spr in spr_runs:
            run_dicts_tr.append(
                build_epoch_dict(unpack_pairs(spr["fine_tune"]["losses"]["train"]))
            )
            run_dicts_val.append(
                build_epoch_dict(unpack_pairs(spr["fine_tune"]["losses"]["val"]))
            )
        ep_tr, mean_tr, sem_tr = aggregate_across_runs(run_dicts_tr)
        ep_val, mean_val, sem_val = aggregate_across_runs(run_dicts_val)

        if ep_tr or ep_val:
            plt.figure()
            if ep_tr:
                plt.plot(ep_tr, mean_tr, label="Train mean", color="tab:orange")
                plt.fill_between(
                    ep_tr,
                    np.array(mean_tr) - np.array(sem_tr),
                    np.array(mean_tr) + np.array(sem_tr),
                    color="tab:orange",
                    alpha=0.3,
                    label="Train ± SEM",
                )
            if ep_val:
                plt.plot(ep_val, mean_val, label="Val mean", color="tab:green")
                plt.fill_between(
                    ep_val,
                    np.array(mean_val) - np.array(sem_val),
                    np.array(mean_val) + np.array(sem_val),
                    color="tab:green",
                    alpha=0.3,
                    label="Val ± SEM",
                )
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("SPR Fine-tune Losses (Aggregated)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "SPR_finetune_loss_agg.png"))
            plt.close()
            print("Aggregated fine-tune loss curve saved.")
    except Exception as e:
        print(f"Error creating aggregated fine-tune loss plot: {e}")
        plt.close()

    # 3. Fine-tune metrics (SWA, CWA, CompWA) -------------------------------------------------------
    try:
        metric_names = ["SWA", "CWA", "CompWA"]
        colors = ["tab:blue", "tab:red", "tab:purple"]
        plt.figure()
        plotted_any = False
        for mname, color in zip(metric_names, colors):
            run_dicts = []
            for spr in spr_runs:
                pairs = unpack_pairs(spr["fine_tune"]["metrics"].get(mname, []))
                run_dicts.append(build_epoch_dict(pairs))
            ep_m, mean_m, sem_m = aggregate_across_runs(run_dicts)
            if ep_m:
                plotted_any = True
                plt.plot(ep_m, mean_m, label=f"{mname} mean", color=color)
                plt.fill_between(
                    ep_m,
                    np.array(mean_m) - np.array(sem_m),
                    np.array(mean_m) + np.array(sem_m),
                    color=color,
                    alpha=0.3,
                    label=f"{mname} ± SEM",
                )
        if plotted_any:
            plt.xlabel("Epoch")
            plt.ylabel("Weighted Accuracy")
            plt.title("SPR Fine-tune Metrics (Aggregated)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "SPR_metrics_agg.png"))
            plt.close()
            print("Aggregated metrics curve saved.")
        else:
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated metrics plot: {e}")
        plt.close()

    # 4. Accuracy per epoch -------------------------------------------------------------------------
    try:
        run_dicts_acc = []
        for spr in spr_runs:
            preds_pair = spr["fine_tune"]["predictions"]
            gts_pair = spr["fine_tune"]["ground_truth"]
            epochs = [ep for ep, _ in preds_pair]
            acc_vals = []
            for (_, p), (_, g) in zip(preds_pair, gts_pair):
                p_arr = np.array(p)
                g_arr = np.array(g)
                acc_vals.append((p_arr == g_arr).mean())
            run_dicts_acc.append({ep: acc for ep, acc in zip(epochs, acc_vals)})
        ep_a, mean_a, sem_a = aggregate_across_runs(run_dicts_acc)

        if ep_a:
            plt.figure()
            plt.plot(ep_a, mean_a, label="Accuracy mean", color="tab:cyan")
            plt.fill_between(
                ep_a,
                np.array(mean_a) - np.array(sem_a),
                np.array(mean_a) + np.array(sem_a),
                color="tab:cyan",
                alpha=0.3,
                label="± SEM",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("SPR Fine-tune Accuracy (Aggregated)")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "SPR_accuracy_agg.png"))
            plt.close()
            print("Aggregated accuracy curve saved.")
    except Exception as e:
        print(f"Error creating aggregated accuracy plot: {e}")
        plt.close()
