import os, pathlib, random, time, math, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------
# working dir and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------------------------------------------
# load SPR_BENCH
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    dset = DatasetDict()
    for split in ["train", "dev", "test"]:
        dset[split] = _load(f"{split}.csv")
    return dset


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)


# ---------------------------------------------------------------------
# utilities for metrics
def count_shape_variety(seq):
    return len(set(tok[0] for tok in seq.strip().split()))


def count_color_variety(seq):
    return len(set(tok[1] for tok in seq.strip().split() if len(tok) > 1))


def comp_weight(seq):
    return count_shape_variety(seq) * count_color_variety(seq)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [comp_weight(s) for s in seqs]
    corr = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(corr) / sum(w) if sum(w) > 0 else 0.0


# ---------------------------------------------------------------------
# build vocab
special_tokens = ["<pad>", "<unk>", "<cls>", "<mask>"]
vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
for s in spr["train"]["sequence"]:
    for tok in s.split():
        if tok not in vocab:
            vocab[tok] = len(vocab)
pad_id, unk_id, cls_id, mask_id = [vocab[t] for t in special_tokens]
print(f"Vocab size = {len(vocab)}")

# labels
labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}
num_classes = len(labels)


# ---------------------------------------------------------------------
# augmentation helpers
def token_ids(tokens):
    return [vocab.get(t, unk_id) for t in tokens]


def mask_shuffle(tokens):
    out = []
    for t in tokens:
        if random.random() < 0.15:
            out.append("<mask>")  # 15 % MLM masking
        else:
            out.append(t)
    # local shuffle (context-aware)
    for i in range(0, len(out), 4):
        window = out[i : i + 4]
        random.shuffle(window)
        out[i : i + 4] = window
    return out


# ---------------------------------------------------------------------
# datasets
class SPRPretrainDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        view1_toks = mask_shuffle(toks.copy())
        view2_toks = mask_shuffle(toks.copy())

        # build MLM labels for view1 (only positions masked become label, others -100)
        mlm_input, mlm_labels = [], []
        for orig, v in zip(toks, view1_toks):
            if v == "<mask>":
                mlm_input.append(mask_id)
                mlm_labels.append(vocab.get(orig, unk_id))
            else:
                mlm_input.append(vocab.get(v, unk_id))
                mlm_labels.append(-100)

        # prepend CLS token
        view1 = [cls_id] + token_ids(view1_toks)
        view2 = [cls_id] + token_ids(view2_toks)
        mlm_inp = [cls_id] + mlm_input
        mlm_lab = [-100] + mlm_labels  # never predict CLS

        return (
            torch.tensor(view1, dtype=torch.long),
            torch.tensor(view2, dtype=torch.long),
            torch.tensor(mlm_inp, dtype=torch.long),
            torch.tensor(mlm_lab, dtype=torch.long),
        )


class SPRCLS(Dataset):
    def __init__(self, seqs, labs):
        self.seqs = seqs
        self.labs = [lab2id[l] for l in labs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [cls_id] + token_ids(self.seqs[idx].split())
        return torch.tensor(ids, dtype=torch.long), self.labs[idx], self.seqs[idx]


def pad_batch(batch_tensors, pad_value=pad_id):
    lengths = [t.size(0) for t in batch_tensors]
    max_len = max(lengths)
    padded = torch.full((len(batch_tensors), max_len), pad_value, dtype=torch.long)
    for i, t in enumerate(batch_tensors):
        padded[i, : t.size(0)] = t
    return padded, torch.tensor(lengths, dtype=torch.long)


def collate_pretrain(batch):
    v1, v2, mlm_inp, mlm_lab = zip(*batch)
    v1_pad, len1 = pad_batch(v1)
    v2_pad, len2 = pad_batch(v2)
    mlm_inp_pad, len_mlm = pad_batch(mlm_inp)
    mlm_lab_pad, _ = pad_batch(mlm_lab, pad_value=-100)
    return (
        v1_pad.to(device),
        len1.to(device),
        v2_pad.to(device),
        len2.to(device),
        mlm_inp_pad.to(device),
        mlm_lab_pad.to(device),
    )


def collate_cls(batch):
    seqs, ys, raw = zip(*batch)
    seq_pad, lens = pad_batch(seqs)
    return (
        seq_pad.to(device),
        lens.to(device),
        torch.tensor(ys, dtype=torch.long).to(device),
        raw,
    )


# ---------------------------------------------------------------------
# model
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, nhead=4, nlayers=2, dim_ff=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(512, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=dim_ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.mlp = nn.Linear(emb_dim, 128)
        # MLM head shares weights
        self.mlm_decoder = nn.Linear(emb_dim, vocab_size, bias=False)
        self.mlm_decoder.weight = self.emb.weight

    def forward(self, x, lengths, project=True):
        # x: B,L
        pos = (
            torch.arange(0, x.size(1), device=x.device)
            .unsqueeze(0)
            .expand(x.size(0), -1)
        )
        h = self.emb(x) + self.pos_emb(pos)
        key_padding_mask = x == pad_id
        out = self.encoder(h, src_key_padding_mask=key_padding_mask)
        cls_rep = out[:, 0]  # CLS token
        if project:
            return self.mlp(cls_rep)  # (B,128)
        else:
            return out, cls_rep  # full sequence & pooled rep


# ---------------------------------------------------------------------
# losses
def nt_xent(z1, z2, temp=0.5):
    z = torch.cat([z1, z2], dim=0)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temp
    N = z1.size(0)
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    targets = torch.arange(N, 2 * N, device=z.device)
    loss1 = nn.functional.cross_entropy(sim[:N], targets)
    loss2 = nn.functional.cross_entropy(sim[N:], targets - N)
    return 0.5 * (loss1 + loss2)


# ---------------------------------------------------------------------
# pre-train
def pretrain_encoder(epochs=2, batch_size=256, lr=1e-3):
    enc = TransformerEncoder(len(vocab)).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    ds = SPRPretrainDataset(spr["train"]["sequence"])
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pretrain
    )
    enc.train()
    for ep in range(1, epochs + 1):
        t0, tot, cnt = time.time(), 0.0, 0
        for (
            v1,
            len1,
            v2,
            len2,
            mlm_inp,
            mlm_lab,
        ) in loader:
            z1 = enc(v1, len1)  # (B,128)
            z2 = enc(v2, len2)
            # MLM on first view
            seq_out, _ = enc(mlm_inp, len1, project=False)
            mlm_logits = enc.mlm_decoder(seq_out)
            mlm_loss = nn.functional.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                mlm_lab.view(-1),
                ignore_index=-100,
            )
            c_loss = nt_xent(z1, z2)
            loss = c_loss + mlm_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * v1.size(0)
            cnt += v1.size(0)
        print(f"Pre-train epoch {ep}: loss={(tot/cnt):.4f}  time={time.time()-t0:.1f}s")
    return enc


pretrained_encoder = pretrain_encoder()

# ---------------------------------------------------------------------
# fine-tuning for classification
train_loader = DataLoader(
    SPRCLS(spr["train"]["sequence"], spr["train"]["label"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_cls,
)
dev_loader = DataLoader(
    SPRCLS(spr["dev"]["sequence"], spr["dev"]["label"]),
    batch_size=512,
    shuffle=False,
    collate_fn=collate_cls,
)


class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder
        self.head = nn.Linear(128, num_classes)

    def forward(self, x, lengths):
        z = self.enc(x, lengths)
        return self.head(z)


model = Classifier(pretrained_encoder).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# experiment data dict
experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

best_val_loss, patience, wait = 1e9, 3, 0
for epoch in range(1, 21):
    # ---- train ----
    model.train()
    tr_loss_sum, seen = 0.0, 0
    for x, lens, y, _ in train_loader:
        optimizer.zero_grad()
        out = model(x, lens)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tr_loss_sum += loss.item() * x.size(0)
        seen += x.size(0)
    tr_loss = tr_loss_sum / seen

    # ---- validation ----
    model.eval()
    val_loss_sum, seen, preds, gts, raws = 0.0, 0, [], [], []
    with torch.no_grad():
        for x, lens, y, raw in dev_loader:
            out = model(x, lens)
            loss = criterion(out, y)
            val_loss_sum += loss.item() * x.size(0)
            seen += x.size(0)
            preds.extend(out.argmax(1).cpu().tolist())
            gts.extend(y.cpu().tolist())
            raws.extend(raw)
    val_loss = val_loss_sum / seen
    compwa = comp_weighted_accuracy(
        raws, [id2lab[i] for i in gts], [id2lab[i] for i in preds]
    )

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(compwa)
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  CompWA = {compwa:.4f}")

    # early stopping
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        wait = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        experiment_data["SPR_BENCH"]["predictions"] = preds
        experiment_data["SPR_BENCH"]["ground_truth"] = gts
    else:
        wait += 1
    if wait >= patience:
        break

# reload best
model.load_state_dict(best_state)

# save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Finished. All metrics saved to ./working/experiment_data.npy")
