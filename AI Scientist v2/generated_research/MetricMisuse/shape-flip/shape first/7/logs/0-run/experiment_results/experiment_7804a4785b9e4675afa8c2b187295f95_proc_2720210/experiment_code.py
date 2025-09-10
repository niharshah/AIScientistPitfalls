import os, random, torch, numpy as np
from datasets import load_dataset, DatasetDict
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# Working dir & device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Store experiment info
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}


# ------------------------------------------------------------------
# Metric helpers
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.split()))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.split()))


def majority_shape_count(seq):
    from collections import Counter

    return Counter(tok[0] for tok in seq.split()).most_common(1)[0][1]


def majority_color_count(seq):
    from collections import Counter

    return Counter(tok[1] for tok in seq.split()).most_common(1)[0][1]


def shape_weighted_accuracy(seqs, y_true, y_pred):
    w = [count_shape_variety(s) for s in seqs]
    correct = [wi if t == p else 0 for wi, t, p in zip(w, y_true, y_pred)]
    return sum(correct) / max(sum(w), 1e-6)


# ------------------------------------------------------------------
# Data loading / synthetic fallback
def load_spr_bench(root: str) -> DatasetDict:
    def _load(name):
        return load_dataset(
            "csv",
            data_files=os.path.join(root, f"{name}.csv"),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(train=_load("train"), dev=_load("dev"), test=_load("test"))


def make_synth(path, n_train=6000, n_dev=1500, n_test=1500):
    shapes, colors = list("STCH"), list("RGBY")

    def rnd():
        L = random.randint(4, 12)
        return " ".join(random.choice(shapes) + random.choice(colors) for _ in range(L))

    def rule(seq):
        return int(count_shape_variety(seq) % 2 == 0)

    os.makedirs(path, exist_ok=True)
    for n, split in [(n_train, "train"), (n_dev, "dev"), (n_test, "test")]:
        with open(os.path.join(path, f"{split}.csv"), "w") as f:
            f.write("id,sequence,label\n")
            for i in range(n):
                s = rnd()
                f.write(f"{i},{s},{rule(s)}\n")


root = "SPR_BENCH"
if not (
    os.path.isdir(root)
    and all(
        os.path.isfile(os.path.join(root, f"{s}.csv")) for s in ["train", "dev", "test"]
    )
):
    print("Real SPR_BENCH not found – creating synthetic set.")
    make_synth(root)
spr = load_spr_bench(root)
print({k: len(v) for k, v in spr.items()})


# ------------------------------------------------------------------
# Vocab (shape+colour tokens)
def build_vocab(ds):
    vocab = {"<pad>": 0, "<unk>": 1}
    for seq in ds["sequence"]:
        for tok in seq.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


vocab = build_vocab(spr["train"])
max_len = 20


def encode(seq):
    ids = [vocab.get(tok, 1) for tok in seq.split()][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


# ------------------------------------------------------------------
class SPRTorch(Dataset):
    def __init__(self, hf):
        self.seqs = hf["sequence"]
        self.labels = hf["label"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        sym = np.array(
            [
                count_shape_variety(s),
                count_color_variety(s),
                len(s.split()),
                majority_shape_count(s),
                majority_color_count(s),
            ],
            dtype=np.float32,
        )
        return {
            "tok": torch.tensor(encode(s), dtype=torch.long),
            "sym": torch.tensor(sym),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "raw": s,
        }


batch_size = 64
train_dl = DataLoader(SPRTorch(spr["train"]), batch_size=batch_size, shuffle=True)
dev_dl = DataLoader(SPRTorch(spr["dev"]), batch_size=batch_size)
test_dl = DataLoader(SPRTorch(spr["test"]), batch_size=batch_size)


# ------------------------------------------------------------------
# Model: gated fusion Transformer
class TokenSplitEmbed(nn.Module):
    def __init__(self, vocab_size, dim=128):
        super().__init__()
        self.base = nn.Embedding(vocab_size, dim, padding_idx=0)
        # optional extra split embeddings (shape & colour) could be added here

    def forward(self, x):
        return self.base(x)


class NeuroSymTransformer(nn.Module):
    def __init__(self, vocab_size, dim=128, nhead=4, layers=2, sym_dim=5, n_cls=2):
        super().__init__()
        self.tok_emb = TokenSplitEmbed(vocab_size, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.neural_fc = nn.Linear(dim, n_cls)
        self.sym_net = nn.Sequential(
            nn.Linear(sym_dim, 32), nn.ReLU(), nn.Linear(32, n_cls)
        )
        self.gate_param = nn.Parameter(torch.tensor(0.0))  # sigmoid -> 0.5 initial

    def forward(self, tok_ids, sym_feats):
        mask = tok_ids == 0
        x = self.tok_emb(tok_ids)
        h = self.transformer(x, src_key_padding_mask=mask)
        pooled = h.mean(1)
        neural_logits = self.neural_fc(pooled)
        sym_logits = self.sym_net(sym_feats)
        alpha = torch.sigmoid(self.gate_param)  # 0..1
        return alpha * neural_logits + (1 - alpha) * sym_logits


model = NeuroSymTransformer(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# ------------------------------------------------------------------
def run_epoch(dl, train=False):
    model.train() if train else model.eval()
    tot_loss, ys, ps, seqs = 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for batch in dl:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            logits = model(batch["tok"], batch["sym"])
            loss = criterion(logits, batch["label"])
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * len(batch["label"])
            pred = logits.argmax(-1).detach().cpu().tolist()
            ys.extend(batch["label"].cpu().tolist())
            ps.extend(pred)
            seqs.extend(batch["raw"])
    return tot_loss / len(dl.dataset), shape_weighted_accuracy(seqs, ys, ps), ys, ps


# ------------------------------------------------------------------
epochs = 10
for ep in range(1, epochs + 1):
    tr_loss, tr_swa, _, _ = run_epoch(train_dl, True)
    dv_loss, dv_swa, _, _ = run_epoch(dev_dl, False)
    experiment_data["SPR_BENCH"]["losses"]["train"].append((ep, tr_loss))
    experiment_data["SPR_BENCH"]["losses"]["val"].append((ep, dv_loss))
    experiment_data["SPR_BENCH"]["metrics"]["train"].append((ep, tr_swa))
    experiment_data["SPR_BENCH"]["metrics"]["val"].append((ep, dv_swa))
    print(f"Epoch {ep}: validation_loss = {dv_loss:.4f}, SWA = {dv_swa:.4f}")

# ------------------------------------------------------------------
_, test_swa, gt, pred = run_epoch(test_dl, False)
print(f"Test SWA = {test_swa:.4f}")
experiment_data["SPR_BENCH"]["predictions"] = pred
experiment_data["SPR_BENCH"]["ground_truth"] = gt

# save
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Experiment data saved to", os.path.join(working_dir, "experiment_data.npy"))
