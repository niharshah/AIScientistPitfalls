import os, pathlib, time, random, json, numpy as np, torch, math, re
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from datasets import load_dataset, DatasetDict

# -----------------------------------------------------------
# compulsory dirs / device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------------------------------------
def load_spr_bench(root: pathlib.Path) -> DatasetDict:
    def _load(csv_name: str):
        return load_dataset(
            "csv",
            data_files=str(root / csv_name),
            split="train",
            cache_dir=".cache_dsets",
        )

    return DatasetDict(
        train=_load("train.csv"),
        dev=_load("dev.csv"),
        test=_load("test.csv"),
    )


SPR_PATH = pathlib.Path(
    os.getenv("SPR_DATA_PATH", "/home/zxl240011/AI-Scientist-v2/SPR_BENCH")
)
spr = load_spr_bench(SPR_PATH)
print({k: len(v) for k, v in spr.items()})

# -----------------------------------------------------------
CLS, PAD, UNK = "[CLS]", "[PAD]", "[UNK]"
vocab = {PAD: 0, CLS: 1, UNK: 2}


def add_token(t):
    if t not in vocab:
        vocab[t] = len(vocab)


for seq in spr["train"]["sequence"]:
    for tok in seq.strip().split():
        add_token(tok)
vocab_size = len(vocab)
label2id = {lab: i for i, lab in enumerate(sorted(set(spr["train"]["label"])))}
id2label = {i: l for l, i in label2id.items()}
num_labels = len(label2id)
print("Vocab size =", vocab_size, " Num labels =", num_labels)

MAX_LEN, BATCH_SIZE, EPOCHS = 128, 64, 5


def encode_sequence(seq: str):
    tokens = [CLS] + seq.strip().split()
    ids = [vocab.get(t, vocab[UNK]) for t in tokens][:MAX_LEN]
    attn = [1] * len(ids)
    if len(ids) < MAX_LEN:
        pad = MAX_LEN - len(ids)
        ids += [vocab[PAD]] * pad
        attn += [0] * pad
    return ids, attn


class SPRDataset(Dataset):
    def __init__(self, split):
        self.seqs = split["sequence"]
        self.labels = split["label"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        ids, attn = encode_sequence(self.seqs[idx])
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(attn),
            "labels": torch.tensor(label2id[self.labels[idx]]),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


train_loader = DataLoader(
    SPRDataset(spr["train"]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
dev_loader = DataLoader(
    SPRDataset(spr["dev"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# -----------------------------------------------------------
class SimpleTransformer(nn.Module):
    def __init__(
        self, vocab_size, num_labels, d_model=128, nhead=4, nlayers=2, dim_ff=256
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=vocab[PAD])
        self.pos = nn.Parameter(torch.zeros(1, MAX_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, ids, attn):
        x = self.embed(ids) + self.pos[:, : ids.size(1)]
        x = self.encoder(x, src_key_padding_mask=~attn.bool())
        return self.cls(x[:, 0])


# -----------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, preds, gts = 0.0, [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits, batch["labels"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tot_loss += loss.item() * batch["labels"].size(0)
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        gts.extend(batch["labels"].cpu().tolist())
    return (
        tot_loss / len(loader.dataset),
        f1_score(gts, preds, average="macro"),
        preds,
        gts,
    )


# -----------------------------------------------------------
NLAYERS_LIST = [1, 2, 3, 4, 6]
experiment_data = {"nlayers_tuning": {"SPR_BENCH": {}}}

for nl in NLAYERS_LIST:
    print(f"\n===== Running experiment with nlayers={nl} =====")
    model = SimpleTransformer(vocab_size, num_labels, nlayers=nl).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    run_dict = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_f1, pred, gt = run_epoch(model, dev_loader, criterion)
        run_dict["metrics"]["train"].append({"epoch": epoch, "macro_f1": tr_f1})
        run_dict["metrics"]["val"].append({"epoch": epoch, "macro_f1": vl_f1})
        run_dict["losses"]["train"].append({"epoch": epoch, "loss": tr_loss})
        run_dict["losses"]["val"].append({"epoch": epoch, "loss": vl_loss})
        print(
            f"Epoch {epoch}: tr_loss={tr_loss:.4f} vl_loss={vl_loss:.4f} tr_F1={tr_f1:.4f} vl_F1={vl_f1:.4f} ({time.time()-t0:.1f}s)"
        )
    run_dict["predictions"] = pred
    run_dict["ground_truth"] = gt
    experiment_data["nlayers_tuning"]["SPR_BENCH"][f"nlayers_{nl}"] = run_dict
    del model
    torch.cuda.empty_cache()

# -----------------------------------------------------------
np.save(
    os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True
)
print("Saved experiment data to", os.path.join(working_dir, "experiment_data.npy"))
