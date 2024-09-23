import numpy as np
from numba import jit, int64, float64


@jit("float64[:](float64[:], float64[:], float64[:], float64[:])", nopython=True, cache=True)
def OT1D_dual(a, b, x, y):
    assert len(x) == len(y)
    n = len(x)

    # init
    A = np.cumsum(a[::-1])[::-1]
    B = np.cumsum(b[::-1])[::-1]
    f = np.zeros(n)

    # C = |x - y|
    i, j = 1, 1
    c = np.abs(x[0] - y[0])
    while i < n and j < n:
        if A[i] >= B[j]:
            f[i] = f[i-1] + np.abs(x[i] - y[j-1]) - c
            c = np.abs(x[i] - y[j-1])
            i = i + 1
        else:
            c = np.abs(x[i-1] - y[j])
            j = j + 1
    while i < n:
        f[i] = f[i-1] + np.abs(x[i] - y[-1]) - c
        c = c + np.abs(x[i] - y[-1])
        i = i + 1
    return f


def update_node_weights(a, g, I, V, lr):
    a[I] = a[I] * np.exp(- lr * V * (g[I] - g[I].min()))
    a[I] = V * a[I] / a[I].sum()
    return a


def fast_stealth_sampling_fixed(X, K, n_slice, eps):
    assert len(X) == len(K)
    
    C = len(X)     
    V = np.array(K)

    Y = np.concatenate(X, axis=0)
    Y = Y + 1e-10 * np.random.randn(*Y.shape)
    N = len(Y)   
    D = len(Y[0])
    X_shapes = np.array([arr.shape[0] for arr in X])
    I = np.split(np.arange(N), np.cumsum(X_shapes)[:-1])

    b = np.array([np.sum(V) / N for i in range(N)])
    a = np.zeros(N)
    for i in range(C):
        a[I[i]] = V[i] / N

    WX = []
    for _ in range(n_slice):
        w = np.random.randn(D)
        w = w / np.linalg.norm(w)
        wx = Y @ w
        idx = np.argsort(wx)
        wx = wx[idx]
        WX.append([wx, idx])
    
    obj = []
    a_cum = np.zeros_like(a)
    a_mean = []
    lr_sum = 0.0
    # iteration
    for itr in range(30000):
        t = np.random.randint(n_slice)
        wx, idx = WX[t]

        # sort
        aSorted, bSorted = a.copy(), b.copy()
        aSorted, bSorted = aSorted[idx], bSorted[idx]

        g = OT1D_dual(aSorted, bSorted, wx, wx)

        idx_inv = np.zeros_like(idx)
        idx_inv[idx] = np.arange(N)
        g = g[idx_inv]

        # learning rate
        lr = 1.0 / np.sqrt(itr + 1) / (3 * np.sum(K))
        lr_sum = lr_sum + lr

        # update
        for i in range(C):
            a = update_node_weights(a, g, I[i], V[i], lr)

        a_cum = a_cum + lr * a
        a_mean.append(a_cum / lr_sum)

        if itr >= 100:
            diff = np.abs(a_mean[-2] - a_mean[-1])
            if np.sum(diff) < eps:
                break
        
    a_mean[-1] = a_mean[-1] / np.sum(a_mean[-1])
    p = np.split(a_mean[-1], np.cumsum(X_shapes))

    return [np.array(p[:-1][i]) for i in range(C)], obj



def fast_stealth_sampling_randomized(X, K, eps):
    assert len(X) == len(K)
    
    C = len(X)     
    V = np.array(K)

    Y = np.concatenate(X, axis=0)
    Y = Y + 1e-10 * np.random.randn(*Y.shape)
    N = len(Y)    
    D = len(Y[0]) 
    X_shapes = np.array([arr.shape[0] for arr in X])
    I = np.split(np.arange(N), np.cumsum(X_shapes)[:-1])

    b = np.array([np.sum(V) / N for i in range(N)])
    a = np.zeros(N)
    for i in range(C):
        a[I[i]] = V[i] / N

    obj = []
    a_cum = np.zeros_like(a)
    a_mean = []
    lr_sum  = 0.0
    # iteration
    for itr in range(30000):
        w = np.random.randn(D)
        w = w / np.linalg.norm(w)
        wx = Y @ w

        # sort
        aSorted, bSorted = a.copy(), b.copy()
        idx = np.argsort(wx)
        wx = wx[idx]
        aSorted, bSorted = aSorted[idx], bSorted[idx]

        g = OT1D_dual(aSorted, bSorted, wx, wx)

        idx_inv = np.zeros_like(idx)
        idx_inv[idx] = np.arange(N)
        g = g[idx_inv]

        # learning rate
        lr = 1.0 / np.sqrt(itr + 1) / (3 * np.sum(K))
        lr_sum = lr_sum + lr

        # update
        for i in range(C):
            a = update_node_weights(a, g, I[i], V[i], lr)

        a_cum = a_cum + lr * a
        a_mean.append(a_cum / lr_sum)
        
        if itr >= 100:
            diff = np.abs(a_mean[-2] - a_mean[-1])
            if np.sum(diff) < eps:
                break
        
    a_mean[-1] = a_mean[-1] / np.sum(a_mean[-1])
    p = np.split(a_mean[-1], np.cumsum(X_shapes))

    return [np.array(p[:-1][i]) for i in range(C)], obj