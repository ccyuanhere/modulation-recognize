import os, json, numpy as np
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from .dataio.radioml_loader import load_radioml2016_dict, split_80_20_like_notebook
from .models.cnn2 import CNN2TFEquivalent
from .utils.plots import plot_training_curves

def make_loaders(X_train, Y_train, X_test, Y_test, batch_size):
    Xtr = torch.from_numpy(X_train).unsqueeze(1)  # [N,1,2,128]
    Xte = torch.from_numpy(X_test).unsqueeze(1)
    ytr = torch.from_numpy(np.argmax(Y_train, axis=1)).long()
    yte = torch.from_numpy(np.argmax(Y_test, axis=1)).long()
    tr_ds = TensorDataset(Xtr, ytr)
    te_ds = TensorDataset(Xte, yte)
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    te = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return tr, te

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 不再需要命令行指定 data_path，默认使用 dataio 目录下的 pickle
    X, lbl, mods, snrs = load_radioml2016_dict()
    (Xtr, Ytr, ytr_int, snr_tr, train_idx), (Xte, Yte, yte_int, snr_te, test_idx) = split_80_20_like_notebook(X, lbl, mods, seed=2016)

    # 可选：对输入做每样本零均值单位方差归一化（环境变量 NORM=1 开启）
    if os.environ.get('NORM', '0') == '1':
        def norm(a):
            m = a.mean(axis=(1,2), keepdims=True)
            s = a.std(axis=(1,2), keepdims=True) + 1e-6
            return (a - m) / s
        Xtr = norm(Xtr)
        Xte = norm(Xte)
        if os.environ.get('DEBUG','0')=='1':
            print('[DEBUG] 应用逐样本归一化: 新范围 train min/max', float(Xtr.min()), float(Xtr.max()))

    # quick overfit 模式：取极小子集验证模型能否记住 (环境变量 OVERFIT=1)
    if os.environ.get('OVERFIT','0')=='1':
        k = min(2048, Xtr.shape[0])
        Xte = Xtr[:k].copy(); Yte = Ytr[:k].copy()
        Xtr = Xtr[:k]; Ytr = Ytr[:k]
        if os.environ.get('DEBUG','0')=='1':
            print(f'[DEBUG] OVERFIT 模式启用, 使用 {k} 样本做 train/val')

    debug = os.environ.get('DEBUG', '0') == '1'
    if debug:
        print("[DEBUG] 训练样本数:", Xtr.shape[0], "测试样本数:", Xte.shape[0])
        print("[DEBUG] 输入张量范围 train min/max:", float(Xtr.min()), float(Xtr.max()))
        uniques, counts = np.unique(np.argmax(Ytr, axis=1), return_counts=True)
        dist = {mods[u]: int(c) for u, c in zip(uniques, counts)}
        print("[DEBUG] 类别分布(训练):", dist)
        print("[DEBUG] 类别数:", len(mods), mods)

    train_loader, val_loader = make_loaders(Xtr, Ytr, Xte, Yte, args.batch_size)

    conv_dropout = float(os.environ.get('CONV_DROPOUT', '0.0'))
    use_xavier = os.environ.get('NO_XAVIER','0') != '1'
    model = CNN2TFEquivalent(num_classes=len(mods), dropout_p=args.dropout, conv_dropout=conv_dropout, use_xavier=use_xavier).to(device)
    if os.environ.get('DEBUG','0')=='1':
        print(f'[DEBUG] 模型参数量: {sum(p.numel() for p in model.parameters())}')
        print(f'[DEBUG] conv_dropout={conv_dropout} use_xavier={use_xavier}')
    criterion = nn.CrossEntropyLoss()
    # Weight Decay & Scheduler 配置
    weight_decay = float(os.environ.get('WEIGHT_DECAY', '1e-4'))  # 默认 1e-4
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    use_scheduler = os.environ.get('NO_SCHEDULER','0')!='1'
    if use_scheduler:
        # ReduceLROnPlateau 依据验证集 loss，patience 可用 LR_PATIENCE 指定
        lr_patience = int(os.environ.get('LR_PATIENCE','3'))
        factor = float(os.environ.get('LR_FACTOR','0.5'))
        min_lr = float(os.environ.get('MIN_LR','1e-6'))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=lr_patience,
                                                         cooldown=0, min_lr=min_lr, verbose=False)
    if os.environ.get('DEBUG','0')=='1':
        print(f"[DEBUG] weight_decay={weight_decay} scheduler={'ON' if use_scheduler else 'OFF'}")

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_state = None
    patience = args.patience
    wait = 0
    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[], lr=[])
    disable_early = os.environ.get('NO_EARLY_STOP','0')=='1' or patience <= 0
    if disable_early and os.environ.get('DEBUG','0')=='1':
        print('[DEBUG] 早停已禁用 (NO_EARLY_STOP=1 或 patience<=0)')

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_losses, tr_preds, tr_trues = [], [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if debug and len(tr_losses) == 0:
                print("[DEBUG] 首个 batch logits 统计 mean/std:", float(logits.mean()), float(logits.std()))
            loss = criterion(logits, yb)
            loss.backward()
            if debug and len(tr_losses) == 0:
                g_mean = []
                for n,p in model.named_parameters():
                    if p.grad is not None:
                        g_mean.append(p.grad.abs().mean().item())
                if g_mean:
                    print("[DEBUG] 首个 batch 平均梯度幅度(均值):", float(np.mean(g_mean)))
            optimizer.step()
            tr_losses.append(loss.item())
            tr_preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
            tr_trues.extend(yb.detach().cpu().numpy().tolist())
        tr_loss = float(np.mean(tr_losses))
        tr_acc = accuracy_score(tr_trues, tr_preds)

        model.eval()
        va_losses, va_preds, va_trues = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_losses.append(loss.item())
                va_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                va_trues.extend(yb.cpu().numpy().tolist())
        va_loss = float(np.mean(va_losses))
        va_acc = accuracy_score(va_trues, va_preds)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if use_scheduler:
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(va_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != prev_lr:
                print(f"    -> LR 降至 {new_lr:.6g} (val_loss plateau)")

        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_state = model.state_dict()
            torch.save(best_state, os.path.join(args.out_dir, 'conv.pt'))
            wait = 0
        else:
            if not disable_early:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                    break

        if debug and epoch == 3:
            # 观察第3轮后输出分布
            model.eval()
            with torch.no_grad():
                xb_dbg, yb_dbg = next(iter(val_loader))
                xb_dbg = xb_dbg.to(device)
                out_dbg = model(xb_dbg)
                probs_dbg = torch.softmax(out_dbg, dim=1).mean(0).cpu().numpy()
                print("[DEBUG] 第3轮验证集平均概率分布(前10个):", probs_dbg[:10])

    plot_training_curves(history, os.path.join(args.out_dir, 'history'))
    print("Training complete. Best checkpoint saved to runs/conv.pt by default.")

def _load_config():
    """加载训练配置：优先级 环境变量 > JSON 文件 > 内置默认。

    支持的环境变量（全部可选）：
      EPOCHS, BATCH_SIZE, DROPOUT, LR, OUT_DIR, PATIENCE
    可选 JSON 文件：`config/train_config.json` 例如：
      {"epochs":120, "batch_size":512}
    未提供的键使用默认值。
    """
    defaults = dict(epochs=100, batch_size=1024, dropout=0.5, lr=1e-3, out_dir='runs', patience=5)
    # JSON 文件
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'train_config.json')
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                file_cfg = json.load(f)
            defaults.update({k: file_cfg[k] for k in defaults.keys() if k in file_cfg})
        except Exception as e:
            print(f"警告: 读取配置文件失败 {e}")
    # 环境变量覆盖
    env_map = {
        'epochs': ('EPOCHS', int),
        'batch_size': ('BATCH_SIZE', int),
        'dropout': ('DROPOUT', float),
        'lr': ('LR', float),
        'out_dir': ('OUT_DIR', str),
        'patience': ('PATIENCE', int),
    }
    for k, (env_key, caster) in env_map.items():
        if env_key in os.environ and os.environ[env_key].strip():
            try:
                defaults[k] = caster(os.environ[env_key])
            except ValueError:
                print(f"警告: 环境变量 {env_key} 解析失败，使用默认 {k}={defaults[k]}")
    return SimpleNamespace(**defaults)


def main():
    args = _load_config()
    print("使用训练配置:")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, dropout={args.dropout}, lr={args.lr}, patience={args.patience}, out_dir={args.out_dir}")
    train(args)

if __name__ == '__main__':
    main()
