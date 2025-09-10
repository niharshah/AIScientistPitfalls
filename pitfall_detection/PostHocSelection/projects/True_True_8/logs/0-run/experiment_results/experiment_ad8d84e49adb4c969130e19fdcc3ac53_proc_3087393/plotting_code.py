import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

saved_figs = []

# iterate through ablation -> dataset hierarchy
for abl_name, abl_blob in experiment_data.items():
    for ds_name, ds_blob in abl_blob.items():
        # ---------- Figure 1: contrastive pre-training loss ----------
        try:
            c_loss = np.array(ds_blob["contrastive_pretrain"]["losses"])
            plt.figure()
            plt.plot(c_loss[:, 0], c_loss[:, 1], marker="o")
            plt.title(f"{ds_name} – Contrastive Pre-training Loss ({abl_name})")
            plt.xlabel("Epoch")
            plt.ylabel("NT-Xent Loss")
            fname = f"{ds_name}_contrastive_loss_{abl_name}.png"
            path = os.path.join(working_dir, fname)
            plt.savefig(path)
            saved_figs.append(path)
            plt.close()
        except Exception as e:
            print(f"Error creating contrastive plot for {ds_name}: {e}")
            plt.close()

        # ---------- Figure 2: fine-tune train/val loss ----------
        try:
            ft_train = np.array(ds_blob["fine_tune"]["losses"]["train"])
            ft_val = np.array(ds_blob["fine_tune"]["losses"]["val"])
            plt.figure()
            plt.plot(ft_train[:, 0], ft_train[:, 1], label="Train")
            plt.plot(ft_val[:, 0], ft_val[:, 1], label="Validation")
            plt.title(f"{ds_name} – Fine-tuning Loss ({abl_name})")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.legend()
            fname = f"{ds_name}_finetune_loss_{abl_name}.png"
            path = os.path.join(working_dir, fname)
            plt.savefig(path)
            saved_figs.append(path)
            plt.close()
        except Exception as e:
            print(f"Error creating fine-tune loss plot for {ds_name}: {e}")
            plt.close()

        # ---------- Figure 3: metric curves ----------
        try:
            swa = np.array(ds_blob["fine_tune"]["metrics"]["SWA"])
            cwa = np.array(ds_blob["fine_tune"]["metrics"]["CWA"])
            comp = np.array(ds_blob["fine_tune"]["metrics"]["CompWA"])
            plt.figure()
            plt.plot(swa[:, 0], swa[:, 1], label="SWA")
            plt.plot(cwa[:, 0], cwa[:, 1], label="CWA")
            plt.plot(comp[:, 0], comp[:, 1], label="CompWA")
            plt.title(f"{ds_name} – Weighted Accuracies ({abl_name})")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            fname = f"{ds_name}_metrics_curve_{abl_name}.png"
            path = os.path.join(working_dir, fname)
            plt.savefig(path)
            saved_figs.append(path)
            plt.close()
        except Exception as e:
            print(f"Error creating metric curve for {ds_name}: {e}")
            plt.close()

        # ---------- Figure 4: final metric comparison ----------
        try:
            labels = ["SWA", "CWA", "CompWA"]
            final_vals = [swa[-1, 1], cwa[-1, 1], comp[-1, 1]]
            plt.figure()
            plt.bar(labels, final_vals, color=["tab:blue", "tab:orange", "tab:green"])
            plt.title(f"{ds_name} – Final Metrics ({abl_name})")
            plt.ylim(0, 1)
            fname = f"{ds_name}_final_metrics_{abl_name}.png"
            path = os.path.join(working_dir, fname)
            plt.savefig(path)
            saved_figs.append(path)
            plt.close()
        except Exception as e:
            print(f"Error creating final metrics bar for {ds_name}: {e}")
            plt.close()

print("Saved figures:")
for p in saved_figs:
    print("  ", p)
