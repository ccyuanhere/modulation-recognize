import os, pickle, numpy as np
from typing import Optional

# 默认数据文件路径（与本文件同目录下）
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), 'RML2016.10a_dict.pkl')

def load_radioml2016_dict(path: Optional[str] = None):
    """加载 RadioML2016.10a 字典格式数据。

    如果未提供路径，将默认使用 `src/dataio/RML2016.10a_dict.pkl`。
    """
    if path is None:
        path = DEFAULT_DATA_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到数据文件: {path}")
    Xd = pickle.load(open(path, "rb"), encoding="latin1")
    keys = list(Xd.keys())
    snrs = sorted(list(set([k[1] for k in keys])))
    mods = sorted(list(set([k[0] for k in keys])))

    X_list = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            arr = Xd[(mod, snr)]
            X_list.append(arr)
            for i in range(arr.shape[0]):
                lbl.append((mod, snr))
    X = np.vstack(X_list)  # [N, 2, 128]
    return X.astype(np.float32), lbl, mods, snrs

def to_onehot(y_int, num_classes):
    yy = np.zeros([len(y_int), num_classes], dtype=np.float32)
    yy[np.arange(len(y_int)), y_int] = 1.0
    return yy

def split_80_20_like_notebook(X, lbl, mods, seed=2016):
    rng = np.random.RandomState(seed)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.8)
    all_idx = np.arange(n_examples)
    train_idx = rng.choice(all_idx, size=n_train, replace=False)
    test_idx = np.array(sorted(list(set(all_idx) - set(train_idx))), dtype=np.int64)

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train_int = np.array([mods.index(lbl[i][0]) for i in train_idx], dtype=np.int64)
    y_test_int  = np.array([mods.index(lbl[i][0]) for i in test_idx], dtype=np.int64)

    Y_train = to_onehot(y_train_int, len(mods))
    Y_test  = to_onehot(y_test_int,  len(mods))

    snr_train = np.array([lbl[i][1] for i in train_idx], dtype=np.int64)
    snr_test  = np.array([lbl[i][1] for i in test_idx], dtype=np.int64)

    return (X_train, Y_train, y_train_int, snr_train, train_idx), (X_test, Y_test, y_test_int, snr_test, test_idx)
