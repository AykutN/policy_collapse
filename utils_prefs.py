import numpy as np

def sample_train_prefs(K=24, m=3, include_corners=True, seed=999):
    """Öğrenme tercih vektörleri üretir.
    K: toplam kaç tercih
    m: amaç sayısı
    include_corners: birim köşeleri dahil et (e_i)
    """
    rng = np.random.default_rng(seed)
    prefs = []
    if include_corners:
        for i in range(m):
            e = np.zeros(m, dtype=np.float32)
            e[i] = 1.0
            prefs.append(e)
    while len(prefs) < K:
        x = rng.random(m)
        x = x / x.sum()
        prefs.append(x.astype(np.float32))
    return np.stack(prefs[:K], axis=0)

def sample_test_prefs(K=24, m=3, include_corners=True, seed=999):
    return sample_train_prefs(K=K, m=m, include_corners=include_corners, seed=seed)
