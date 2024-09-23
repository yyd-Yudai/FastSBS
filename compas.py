import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import pathlib
import argparse

import utils
import stealth_sampling
import fast_stealth_sampling
import tree_stealth_sampling



def preprocess_and_save(Nref=1278, seed=0):

    # outdir
    p = pathlib.Path('./compas').joinpath('data')
    p.mkdir(parents=True, exist_ok=True)

    # load data
    with open('compas.pkl', 'rb') as f:       
        d = pickle.load(f)

    X, S, Y = d['data']
    Xbase, Xref, Sbase, Sref, Ybase, Yref = train_test_split(X, S, Y, test_size=Nref, random_state=seed)

    scaler = StandardScaler()
    scaler.fit(Xbase)
    Xbase = scaler.transform(Xbase)
    Xref = scaler.transform(Xref)

    # save
    with open(p.joinpath('compas_%03d.pkl' % (seed,)), 'wb') as f:
        d = {'base':[Xbase, Sbase, Ybase], 'ref':[Xref, Sref, Yref]}
        pickle.dump(d, f)

    
def sampling(Nsample=1000, n_slice=10, eps=1e-2, num_clusters=5, max_depth=5, seed=0):
    p = pathlib.Path('./compas').joinpath('data')
    with open(p.joinpath('compas_%03d.pkl' % (seed,)), 'rb') as f:
        d = pickle.load(f)
        Xbase, Sbase, Ybase = d['base']
        Xref, Sref, _ = d['ref']
        Z = utils.split_to_four(Xbase, Sbase, Ybase)

        # outdir
        p = pathlib.Path('./compas').joinpath('sampling')
        p.mkdir(parents=True, exist_ok=True)

        # sampling
        alphas = np.linspace(0.4, 0.8, 5)
        methods = ['case-control', 'stealth', 'sliced', 'tree']
        res = {alpha:{method:[] for method in methods} for alpha in alphas}
        sampled_spos = np.mean(Sbase)
        for alpha in alphas:
            K = utils.computeK(Z, Nsample, sampled_spos, [alpha, alpha])
            for i, method in enumerate(methods):
                np.random.seed(seed+i)
                if method == 'case-control':
                    q = utils.case_control_sampling([z[:, :-1] for z in Z], K)
                elif method == 'stealth':
                    q, _ = stealth_sampling.stealth_sampling([z[:, :-1] for z in Z], K, path='./', prefix='stealth', timeout=30.0)
                elif method == 'sliced':
                    if n_slice != -1:
                        q, _ = fast_stealth_sampling.fast_stealth_sampling_fixed([z[:, :-1] for z in Z], K, n_slice, eps)
                    else:
                        q, _ = fast_stealth_sampling.fast_stealth_sampling_randomized([z[:, :-1] for z in Z], K, eps)
                elif method == 'tree':
                    if n_slice != -1:
                        q, _ = tree_stealth_sampling.tree_stealth_sampling_fixed([z[:, :-1] for z in Z], K, num_clusters, max_depth, n_slice, eps)
                    else:
                        q, _ = tree_stealth_sampling.tree_stealth_sampling_randomized([z[:, :-1] for z in Z], K, num_clusters, max_depth, eps)

                idx = np.random.choice(Xbase.shape[0], sum(K), p=np.concatenate(q), replace=False)
                res[alpha][method] = (q, idx)

            # save
            with open(p.joinpath('compas_%03d.pkl' % (seed,)), 'wb') as f:
                pickle.dump(res, f)


def evaluate(Nsample, seed=0):

    # outdir
    pout = pathlib.Path('./compas').joinpath('eval')
    pout.mkdir(parents=True, exist_ok=True)

    # load data
    p = pathlib.Path('./compas').joinpath('data')
    with open(p.joinpath('compas_%03d.pkl' % (seed,)), 'rb') as f:
        d = pickle.load(f)
    Xbase, Sbase, Ybase = d['base']
    Xref, Sref, _ = d['ref']
    Z = utils.split_to_four(Xbase, Sbase, Ybase)

    np.random.seed(seed)
    idx = np.random.permutation(Xbase.shape[0])[:Nsample]
    dx, dx_s1, dx_s0 = utils.compute_wasserstein(Xbase[idx, :], Sbase[idx], Xref, Sref, timeout=10.0)
    parity = utils.demographic_parity([z[:, -1] for z in Z])
    results = [parity, dx, dx_s1, dx_s0]

    # save
    with open(pout.joinpath('compas_baseline_%03d.pkl' % (seed,)), 'wb') as f:
        pickle.dump(results, f)

    # load sampling
    p = pathlib.Path('./compas').joinpath('sampling')
    with open(p.joinpath('compas_%03d.pkl' % (seed,)), 'rb') as f:
        res = pickle.load(f)
    
    # evaluate
    c = 0
    for alpha in res.keys():
        results = {m:[] for m in res[alpha].keys()}
        for m in res[alpha].keys():
            _, idx = res[alpha][m]
            Xs = np.concatenate([z[:, :-2] for z in Z], axis=0)[idx, :]
            Ss = np.concatenate([z[:, -2] for z in Z], axis=0)[idx]
            Ts = np.concatenate([z[:, -1] for z in Z], axis=0)[idx]
            Zs = utils.split_to_four(Xs, Ss, Ts)
            parity = utils.demographic_parity([z[:, -1] for z in Zs])
            dx, dx_s1, dx_s0 = utils.compute_wasserstein(Xs, Ss, Xref, Sref, timeout=10.0)
            results[m] = [parity, dx, dx_s1, dx_s0]
            c = c + 1    
        # save
        with open(pout.joinpath('compas_alpha%0.2f_%03d.pkl' % (alpha, seed)), 'wb') as f:
            pickle.dump(results, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['preprocess', 'sampling', 'evaluate'])
    parser.add_argument('--Nref', type=int, default=1278)
    parser.add_argument('--Nsample', type=int, default=1000)
    parser.add_argument('--n_parallel', type=int, default=-1)
    parser.add_argument('--eps', type=float, default=1e-2)
    parser.add_argument('--n_slice', type=int, default=10)
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    args = parser.parse_args()

    if args.mode == 'preprocess':
        fn = lambda seed: preprocess_and_save(args.Nref, seed)
        joblib.Parallel(n_jobs=args.n_parallel)(joblib.delayed(fn)(seed) for seed in range(args.start, args.end))
    elif args.mode == 'sampling':
        for seed in range(args.start, args.end):
            sampling(args.Nsample, args.n_slice, args.eps, args.num_clusters, args.max_depth, seed)
    elif args.mode == 'evaluate':
        for seed in range(args.start, args.end):
            evaluate(args.Nsample, seed)


