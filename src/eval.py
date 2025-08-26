import os, argparse, numpy as np, pickle
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from .dataio.radioml_loader import load_radioml2016_dict, split_80_20_like_notebook
from .models.cnn2 import CNN2TFEquivalent
from .utils.plots import plot_confusion_matrix, plot_acc_vs_snr

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits = []
    all_trues = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        all_logits.append(logits.cpu().numpy())
        all_trues.append(yb.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_trues, axis=0)
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    return y_true, y_pred, probs

def main():
    p = argparse.ArgumentParser()
    # data_path 移除，默认路径由加载器处理
    p.add_argument('--ckpt_path', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--out_dir', type=str, default='runs')
    p.add_argument('--dropout', type=float, default=0.5)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, lbl, mods, snrs = load_radioml2016_dict()
    (Xtr, Ytr, ytr_int, snr_tr, train_idx), (Xte, Yte, yte_int, snr_te, test_idx) = split_80_20_like_notebook(X, lbl, mods, seed=2016)

    Xte_t = torch.from_numpy(Xte).unsqueeze(1)   # [N,1,2,128]
    yte_t = torch.from_numpy(np.argmax(Yte, axis=1)).long()
    te_ds = TensorDataset(Xte_t, yte_t)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = CNN2TFEquivalent(num_classes=len(mods), dropout_p=args.dropout).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    y_true, y_pred, probs = predict(model, te_loader, device)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(mods))))
    plot_confusion_matrix(cm, title='ConvNet Confusion Matrix (Overall)', labels=mods,
                          save_path=os.path.join(args.out_dir, 'confusion_overall.png'), normalize=True)

    test_snrs = np.array([lbl[i][1] for i in test_idx], dtype=np.int64)
    acc = {}
    for snr in snrs:
        mask = (test_snrs == snr)
        if mask.sum() == 0:
            continue
        y_true_s = y_true[mask]
        y_pred_s = y_pred[mask]
        cm_s = confusion_matrix(y_true_s, y_pred_s, labels=list(range(len(mods))))
        plot_confusion_matrix(cm_s, title=f'ConvNet Confusion Matrix (SNR={snr})', labels=mods,
                              save_path=os.path.join(args.out_dir, f'confusion_snr_{snr}.png'), normalize=True)
        cor = np.trace(cm_s)
        ncor = cm_s.sum() - cor
        acc[snr] = float(cor / (cor + ncor))

    plot_acc_vs_snr(snrs, acc, os.path.join(args.out_dir, 'acc_vs_snr.png'))
    with open(os.path.join(args.out_dir, 'results_cnn.pkl'), 'wb') as f:
        pickle.dump(("CNN2", args.dropout, acc), f)

    print("Done. Saved confusion matrices, acc_vs_snr plot, and results_cnn.pkl.")

if __name__ == '__main__':
    main()
