import numpy as np

from matplotlib import pyplot as plt
import stealth_sampling


# split data to bins (s, y) = (1, 1), (1, 0), (0, 1), (0, 0)
def split_to_four(X, S, Y):
    Z = np.c_[X, S, Y]
    Z_pos_pos = Z[np.logical_and(S, Y), :]
    Z_pos_neg = Z[np.logical_and(S, np.logical_not(Y)), :]
    Z_neg_pos = Z[np.logical_and(np.logical_not(S), Y), :]
    Z_neg_neg = Z[np.logical_and(np.logical_not(S), np.logical_not(Y)), :]
    Z = [Z_pos_pos, Z_pos_neg, Z_neg_pos, Z_neg_neg]
    return Z


# compute demographic parity
def demographic_parity(W):
    p_pos = np.mean(np.concatenate(W[:2]))
    p_neg = np.mean(np.concatenate(W[2:]))
    return np.abs(p_pos - p_neg)


# compute the sampling size from each bin
def computeK(Z, Nsample, sampled_spos, sampled_ypos):
    Kpp = Nsample*sampled_spos*sampled_ypos[0]
    Kpn = Nsample*sampled_spos*(1-sampled_ypos[0])
    Knp = Nsample*(1-sampled_spos)*sampled_ypos[1]
    Knn = Nsample*(1-sampled_spos)*(1-sampled_ypos[1])
    K = [Kpp, Kpn, Knp, Knn]
    kratio = min([min(1, z.shape[0]/k) for (z, k) in zip(Z, K)])
    Kpp = int(np.floor(Nsample*kratio*sampled_spos*sampled_ypos[0]))
    Kpn = int(np.floor(Nsample*kratio*sampled_spos*(1-sampled_ypos[0])))
    Knp = int(np.floor(Nsample*kratio*(1-sampled_spos)*sampled_ypos[1]))
    Knn = int(np.floor(Nsample*kratio*(1-sampled_spos)*(1-sampled_ypos[1])))
    K = [max([k, 1]) for k in [Kpp, Kpn, Knp, Knn]]
    return K


# case-contrl sampling
def case_control_sampling(X, K):
    q = [(K[i]/sum(K)) * np.ones(x.shape[0]) / x.shape[0] for i, x in enumerate(X)]
    return q


def gen_data(N, d, spos=0.5, ypos_coef=0.2, seed=0):
    np.random.seed(seed)
    X = np.random.rand(N, d)
    S = (np.random.rand(N) < spos)
    Y = X[:, 0] + ypos_coef * S > 0.5
    return X, S, Y


# compute wasserstein distance
def compute_wasserstein(X1, S1, X2, S2, timeout=60.0):
    while True:
        try:
            dx = stealth_sampling.compute_wasserstein(X1, X2, path='./', prefix='compas', timeout=timeout)
            break
        except:
            pass
    while True:
        try:
            dx_s1 = stealth_sampling.compute_wasserstein(X1[S1>0.5, :], X2[S2>0.5, :], path='./', prefix='compas', timeout=timeout)
            break
        except:
            pass
    while True:
        try:
            dx_s0 = stealth_sampling.compute_wasserstein(X1[S1<0.5, :], X2[S2<0.5, :], path='./', prefix='compas', timeout=timeout)
            break
        except:
            pass
    return dx, dx_s1, dx_s0


# compute wasserstein distance w/ boostrap
def compute_wasserstein_bootstrap(X1, S1, X2, S2, n, prefix, num_sample=5, num_process=10, seed=0):
    while True:
        try:
            dx = stealth_sampling.compute_wasserstein_bootstrap(X1, X2, n, path='./', prefix=prefix+'_1', num_sample=num_sample, num_process=num_process, seed=seed, timeout=60)
            break
        except:
            pass
    while True:
        try:
            dx_s1 = stealth_sampling.compute_wasserstein_bootstrap(X1[S1>0.5, :], X2[S2>0.5, :], n, path='./', prefix=prefix+'_2', num_sample=num_sample, num_process=num_process, seed=seed+1, timeout=60)
            break
        except:
            pass
    while True:
        try:
            dx_s0 = stealth_sampling.compute_wasserstein_bootstrap(X1[S1<0.5, :], X2[S2<0.5, :], n, path='./', prefix=prefix+'_3', num_sample=num_sample, num_process=num_process, seed=seed+2, timeout=60)
            break
        except:
            pass
    return dx, dx_s1, dx_s0
