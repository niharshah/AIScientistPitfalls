import os, random, time, math, numpy as np, torch, pathlib
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict

# ---------------------------------- working dir / device ----------------------------
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ---------------------------------- load SPR ---------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _ld(name):
        return load_dataset(
            "csv", data_files=str(root / name), split="train", cache_dir=".cache_dsets"
        )

    return DatasetDict(train=_ld("train.csv"), dev=_ld("dev.csv"), test=_ld("test.csv"))


DATA_PATH = pathlib.Path("/home/zxl240011/AI-Scientist-v2/SPR_BENCH/")
spr = load_spr_bench(DATA_PATH)


# ---------------------------------- metrics ----------------------------------------
def count_shape(seq):
    return len({tok[0] for tok in seq.split() if tok})


def count_color(seq):
    return len({tok[1] for tok in seq.split() if len(tok) > 1})


def comp_weight(seq):
    return count_shape(seq) * count_color(seq)


def comp_weighted_accuracy(seqs, y_true, y_pred):
    w = [comp_weight(s) for s in seqs]
    good = [wt if t == p else 0 for wt, t, p in zip(w, y_true, y_pred)]
    return sum(good) / sum(w) if sum(w) else 0.0


# ---------------------------------- vocab / labels ---------------------------------
special = ["<pad>", "<unk>", "<cls>", "<mask>"]
vocab = {tok: i for i, tok in enumerate(special)}
for s in spr["train"]["sequence"]:
    for t in s.split():
        if t not in vocab:
            vocab[t] = len(vocab)
pad_id, unk_id, cls_id, mask_id = (
    vocab["<pad>"],
    vocab["<unk>"],
    vocab["<cls>"],
    vocab["<mask>"],
)

labels = sorted(set(spr["train"]["label"]))
lab2id = {l: i for i, l in enumerate(labels)}
id2lab = {i: l for l, i in lab2id.items()}


# ---------------------------------- augmentation -----------------------------------
def shuffle_window(tokens, k=3):
    for i in range(0, len(tokens), k):
        window = tokens[i : i + k]
        random.shuffle(window)
        tokens[i : i + k] = window
    return tokens


def augment(tokens):
    out = tokens.copy()
    out = [
        t if random.random() > 0.3 else random.choice([unk_id, mask_id]) for t in out
    ]
    out = shuffle_window(out, 3)
    return out


def make_masked(tokens, p=0.15):
    labels = []
    out = []
    for t in tokens:
        if random.random() < p:
            labels.append(t)
            out.append(mask_id)
        else:
            labels.append(-100)  # ignored position
            out.append(t)
    return out, labels


# ---------------------------------- datasets ---------------------------------------
class SPRPretrainDS(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences

    def encode(self, toks):  # map to ids
        return [vocab.get(t, unk_id) for t in toks]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx].split()
        ids = self.encode(toks)
        view1 = [cls_id] + augment(ids)
        view2 = [cls_id] + augment(ids)
        mlm_in, mlm_lab = make_masked(ids)
        mlm_in = [cls_id] + mlm_in
        mlm_lab = [-100] + mlm_lab  # CLS not predicted
        return view1, view2, mlm_in, mlm_lab


class SPRClsDS(Dataset):
    def __init__(self, seqs, labs):
        self.seqs = seqs
        self.labs = [lab2id[l] for l in labs]

    def encode(self, toks):
        return [vocab.get(t, unk_id) for t in toks]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids = [cls_id] + self.encode(self.seqs[idx].split())
        return ids, self.labs[idx], self.seqs[idx]


def pad_and_mask(batch_seqs):
    lens = [len(s) for s in batch_seqs]
    m = max(lens)
    arr = np.full((len(batch_seqs), m), pad_id, np.int64)
    for i, s in enumerate(batch_seqs):
        arr[i, : len(s)] = s
    pad_mask = arr == pad_id
    return torch.tensor(arr), torch.tensor(pad_mask)


def collate_pretrain(b):
    v1, v2, m_in, m_lab = zip(*b)
    a, a_mask = pad_and_mask(v1)
    b_, b_mask = pad_and_mask(v2)
    c, c_mask = pad_and_mask(m_in)
    labs = pad_sequences(m_lab, -100)
    return (a, a_mask, b_, b_mask, c, c_mask, labs)


def pad_sequences(seq_list, fill):
    l = [len(s) for s in seq_list]
    m = max(l)
    arr = np.full((len(seq_list), m), fill, np.int64)
    for i, s in enumerate(seq_list):
        arr[i, : len(s)] = s
    return torch.tensor(arr)


def collate_cls(b):
    ids, ys, raw = zip(*b)
    x, mask = pad_and_mask(ids)
    return (x, mask, torch.tensor(ys)), list(raw)


# ---------------------------------- model ------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # B,L,D
        return x + self.pe[:, : x.size(1)]


class Encoder(nn.Module):
    def __init__(self, vocab_sz, d=128, nhead=4, nlayers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, d, padding_idx=pad_id)
        self.pos = PositionalEncoding(d)
        encoder_layer = nn.TransformerEncoderLayer(
            d, nhead, dim_feedforward=256, activation="gelu"
        )
        self.tr = nn.TransformerEncoder(encoder_layer, nlayers)
        self.d = d

    def forward(self, x, pad_mask):  # x:B,L
        e = self.emb(x) * math.sqrt(self.d)
        e = self.pos(e)
        out = self.tr(e.transpose(0, 1), src_key_padding_mask=pad_mask).transpose(0, 1)
        cls = out[:, 0]  # CLS token
        return cls, out  # sequence out for MLM


class SPRModelPretrain(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.proj = nn.Linear(enc.d, 128)
        self.mlm_head = nn.Linear(enc.d, len(vocab))

    def forward(self, x, mask):
        cls, seq = self.enc(x, mask)
        return nn.functional.normalize(self.proj(cls), dim=1), self.mlm_head(seq)


class Classifier(nn.Module):
    def __init__(self, enc, ncls):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(enc.d, ncls)

    def forward(self, x, mask):
        cls, _ = self.enc(x, mask)
        return self.head(cls)


# ---------------------------------- losses -----------------------------------------
def nt_xent(z1, z2, temp=0.5):
    z = torch.cat([z1, z2], 0)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temp
    N = z1.size(0)
    diag = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(diag, -9e15)
    targets = torch.arange(N, 2 * N, device=z.device)
    loss = nn.functional.cross_entropy(sim[:N], targets) + nn.functional.cross_entropy(
        sim[N:], targets - N
    )
    return loss * 0.5


# ---------------------------------- pre-training -----------------------------------
def pretrain_encoder(epochs=2, bs=256, lr=1e-3):
    enc = Encoder(len(vocab)).to(device)
    model = SPRModelPretrain(enc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        SPRPretrainDS(spr["train"]["sequence"]),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_pretrain,
    )
    model.train()
    for ep in range(1, epochs + 1):
        ep_loss, st = 0, time.time()
        for a, a_m, b, b_m, c, c_m, lab in loader:
            a, a_m, b, b_m, c, c_m, lab = [
                t.to(device) for t in (a, a_m, b, b_m, c, c_m, lab)
            ]
            z1, _ = model(a, a_m)
            z2, _ = model(b, b_m)
            _, logit = model(c, c_m)
            loss_con = nt_xent(z1, z2)
            mlm_loss = nn.functional.cross_entropy(
                logit.view(-1, logit.size(-1)), lab.view(-1), ignore_index=-100
            )
            loss = loss_con + 0.5 * mlm_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * a.size(0)
        print(
            f"Pretrain epoch {ep}: loss={ep_loss/len(loader.dataset):.4f} "
            f"time={time.time()-st:.1f}s"
        )
    return enc


pretrained_enc = pretrain_encoder()

# ---------------------------------- fine-tune --------------------------------------
train_loader = DataLoader(
    SPRClsDS(spr["train"]["sequence"], spr["train"]["label"]),
    batch_size=256,
    shuffle=True,
    collate_fn=collate_cls,
)
dev_loader = DataLoader(
    SPRClsDS(spr["dev"]["sequence"], spr["dev"]["label"]),
    batch_size=512,
    shuffle=False,
    collate_fn=collate_cls,
)

model = Classifier(pretrained_enc, len(labels)).to(device)
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

experiment_data = {
    "SPR_BENCH": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

best, wait, patience = 1e9, 0, 3
for epoch in range(1, 16):
    # train
    model.train()
    tr_loss = 0
    for (x, mask, y), _ in train_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        opt.zero_grad()
        out = model(x, mask)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        tr_loss += loss.item() * x.size(0)
    tr_loss /= len(train_loader.dataset)

    # dev
    model.eval()
    val_loss = 0
    preds = []
    gts = []
    raw = []
    with torch.no_grad():
        for (x, mask, y), r in dev_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            o = model(x, mask)
            l = crit(o, y)
            val_loss += l.item() * x.size(0)
            preds.extend(o.argmax(1).cpu().tolist())
            gts.extend(y.cpu().tolist())
            raw.extend(r)
    val_loss /= len(dev_loader.dataset)
    comp = comp_weighted_accuracy(
        raw, [id2lab[i] for i in gts], [id2lab[i] for i in preds]
    )

    experiment_data["SPR_BENCH"]["losses"]["train"].append(tr_loss)
    experiment_data["SPR_BENCH"]["losses"]["val"].append(val_loss)
    experiment_data["SPR_BENCH"]["metrics"]["val"].append(comp)

    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}  CompWA = {comp:.4f}")

    if val_loss < best - 1e-4:
        best = val_loss
        wait = 0
        best_state = model.state_dict()
        experiment_data["SPR_BENCH"]["predictions"] = preds
        experiment_data["SPR_BENCH"]["ground_truth"] = gts
    else:
        wait += 1
    if wait >= patience:
        break

model.load_state_dict(best_state)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print("Training finished – logged to ./working/experiment_data.npy")
