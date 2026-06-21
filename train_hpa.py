"""
HPA Multi-label Classification — Training Script
Logs per-epoch metrics to logs/train_log.jsonl
Reads logs/live_config.json each epoch for live LR override
"""
import os, sys, json, signal
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import timm
    USE_TIMM = True
except ImportError:
    import torchvision.models as tv_models
    USE_TIMM = False

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path('/home/hnguye24/morphogene')
TRAIN_D  = BASE / 'train'
CSV      = BASE / 'train.csv'
LOG_D    = BASE / 'logs'
CKPT_D   = LOG_D / 'checkpoints'
METRICS  = LOG_D / 'train_log.jsonl'
STATUS   = LOG_D / 'train_status.json'
LIVE_CFG = LOG_D / 'live_config.json'
LOG_D.mkdir(exist_ok=True); CKPT_D.mkdir(exist_ok=True)

NC = 28
CHS = ['red', 'green', 'blue', 'yellow']

# ── Dataset ───────────────────────────────────────────────────────────────────
class HPADataset(Dataset):
    def __init__(self, df, size=224, augment=False):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.aug  = augment

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs = []
        for ch in CHS:
            p = TRAIN_D / f"{row['Id']}_{ch}.png"
            a = np.array(Image.open(p).convert('L').resize((self.size, self.size)), dtype=np.float32) / 255.0
            imgs.append(a)
        x = torch.tensor(np.stack(imgs), dtype=torch.float32)
        if self.aug:
            if torch.rand(1) > 0.5: x = torch.flip(x, [2])
            if torch.rand(1) > 0.5: x = torch.flip(x, [1])
            x = torch.rot90(x, torch.randint(0,4,(1,)).item(), [1,2])
        lbl = torch.zeros(NC, dtype=torch.float32)
        for l in str(row['Target']).split(): lbl[int(l)] = 1.0
        return x, lbl

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(name, pretrained):
    if USE_TIMM:
        return timm.create_model(name, pretrained=pretrained, num_classes=NC, in_chans=4)
    m = tv_models.resnet34(pretrained=pretrained)
    oc = m.conv1
    m.conv1 = nn.Conv2d(4, oc.out_channels, oc.kernel_size, oc.stride, oc.padding, bias=False)
    with torch.no_grad():
        m.conv1.weight[:,:3] = oc.weight
        m.conv1.weight[:,3]  = oc.weight[:,0]
    m.fc = nn.Linear(m.fc.in_features, NC)
    return m

# ── Metrics (no sklearn) ──────────────────────────────────────────────────────
def f1_per_class(preds, targets):
    f1s = []
    for c in range(NC):
        tp = ((preds[:,c]==1)&(targets[:,c]==1)).sum()
        fp = ((preds[:,c]==1)&(targets[:,c]==0)).sum()
        fn = ((preds[:,c]==0)&(targets[:,c]==1)).sum()
        denom = 2*tp + fp + fn
        f1s.append(float(2*tp / denom) if denom > 0 else 0.0)
    return f1s

# ── Logging ───────────────────────────────────────────────────────────────────
def write_status(**kw):
    with open(STATUS,'w') as f: json.dump({'ts': datetime.now().isoformat(), **kw}, f)

def log_metrics(m):
    with open(METRICS,'a') as f: f.write(json.dumps(m)+'\n')

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg_path = LOG_D / 'train_config.json'
    with open(cfg_path) as f: cfg = json.load(f)

    model_name  = cfg.get('model', 'efficientnet_b0')
    lr          = float(cfg.get('lr', 1e-4))
    bs          = int(cfg.get('batch_size', 32))
    epochs      = int(cfg.get('epochs', 30))
    sz          = int(cfg.get('img_size', 224))
    val_split   = float(cfg.get('val_split', 0.1))
    pretrained  = bool(cfg.get('pretrained', True))
    nw          = int(cfg.get('num_workers', 4))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Model: {model_name}  LR: {lr}  BS: {bs}")

    df = pd.read_csv(CSV)
    try:
        from sklearn.model_selection import train_test_split as tts
        tr, va = tts(df, test_size=val_split, random_state=42)
    except Exception:
        n = int(len(df)*val_split)
        idx = np.random.RandomState(42).permutation(len(df))
        va, tr = df.iloc[idx[:n]], df.iloc[idx[n:]]

    train_ds = HPADataset(tr, sz, augment=True)
    val_ds   = HPADataset(va, sz, augment=False)
    train_dl = DataLoader(train_ds, bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_dl   = DataLoader(val_ds,   bs, shuffle=False, num_workers=nw, pin_memory=True)

    # class weights
    counts = np.zeros(NC)
    for t in df['Target']:
        for l in str(t).split(): counts[int(l)] += 1
    pos_w = torch.tensor((len(df)-counts)/(counts+1e-6), dtype=torch.float32).to(device)

    model     = build_model(model_name, pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0.0
    if METRICS.exists(): METRICS.unlink()
    write_status(status='running', epoch=0, total=epochs, pid=os.getpid(),
                 model=model_name, device=str(device), best_f1=0.0)

    for ep in range(1, epochs+1):
        write_status(status='running', epoch=ep, total=epochs, pid=os.getpid(),
                     model=model_name, device=str(device), best_f1=round(best_f1,4))

        # live LR override
        if LIVE_CFG.exists():
            try:
                live = json.load(open(LIVE_CFG))
                if 'lr' in live:
                    for pg in optimizer.param_groups: pg['lr'] = float(live['lr'])
                    print(f"  [live] lr → {live['lr']}")
                if 'bs' in live:
                    print(f"  [live] batch_size change takes effect next restart")
                LIVE_CFG.unlink()
            except: pass

        # train
        model.train(); tl = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            tl += loss.item()*len(xb)
        tl /= len(train_ds)

        # val
        model.eval(); vl = 0.0; all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += criterion(out, yb).item()*len(xb)
                all_p.append(torch.sigmoid(out).cpu().numpy())
                all_t.append(yb.cpu().numpy())
        vl /= len(val_ds)
        P = (np.vstack(all_p) > 0.5).astype(int)
        T = np.vstack(all_t).astype(int)
        pcf1 = f1_per_class(P, T)
        mf1  = float(np.mean(pcf1))

        scheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']

        m = dict(epoch=ep, train_loss=round(tl,6), val_loss=round(vl,6),
                 macro_f1=round(mf1,6), per_class_f1=[round(v,4) for v in pcf1],
                 lr=cur_lr, ts=datetime.now().isoformat())
        log_metrics(m)
        print(f"Ep {ep}/{epochs}  tl={tl:.4f}  vl={vl:.4f}  f1={mf1:.4f}  lr={cur_lr:.2e}")

        if mf1 > best_f1:
            best_f1 = mf1
            for old in CKPT_D.glob('best_ep*.pt'): old.unlink()
            torch.save({'epoch':ep,'state':model.state_dict(),'macro_f1':mf1,'cfg':cfg},
                       CKPT_D/f'best_ep{ep:03d}_f1{mf1:.4f}.pt')
        torch.save({'epoch':ep,'state':model.state_dict(),'macro_f1':mf1,'cfg':cfg},
                   CKPT_D/'latest.pt')

    write_status(status='done', epoch=epochs, total=epochs, pid=os.getpid(),
                 model=model_name, device=str(device), best_f1=round(best_f1,4))
    print(f"\nDone. Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
