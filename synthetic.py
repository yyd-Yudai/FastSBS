import numpy as np
from scipy.stats import ks_2samp
import pickle
import joblib
import pathlib
import argparse

import utils
import stealth_sampling
import fast_stealth_sampling
import tree_stealth_sampling


def preprocess_and_save(N=1000, Nref=200, d=1, spos=0.5, ypos_coef=0.2, seed=0):
    # outdir
    p = pathlib.Path('./synthetic').joinpath('data')
    p.mkdir(parents=True, exist_ok=True)

    # data generation
    X, S, Y = utils.gen_data(N, d, spos=spos, ypos_coef=ypos_coef, seed=seed)
    Xref, Sref, Yref = utils.gen_data(Nref, d, spos=spos, ypos_coef=ypos_coef, seed=seed+1)

    # save
    with open(p.joinpath('synthetic_%03d.pkl' % (seed,)), 'wb') as f:
        d = {'te':[X, S, Y], 'ref':[Xref, Sref, Yref]}
        pickle.dump(d, f)

    
def sampling(N=1000, Nsample=200, sampled_spos=0.5, n_slice=10, eps=1e-2, num_clusters=3, max_depth=5, seed=0):
    p = pathlib.Path('./synthetic').joinpath('data')
    with open(p.joinpath('synthetic_%03d.pkl' % (seed,)), 'rb') as f:
        d = pickle.load(f)
        X, S, Y = d['te']
        Z = utils.split_to_four(X, S, Y)

        # outdir
        p = pathlib.Path('./synthetic').joinpath('sampling')
        p.mkdir(parents=True, exist_ok=True)

        # sampling
        alphas = np.linspace(0.4, 0.8, 5)
        methods = ['case-control', 'stealth', 'sliced', 'tree']
        res = {alpha:{m:[] for m in methods} for alpha in alphas}
        for alpha in alphas:
            K = utils.computeK(Z, Nsample, sampled_spos, [alpha, alpha])
            for i, method in enumerate(methods):
                np.random.seed(seed+i)
                if method == 'case-control':
                    q = utils.case_control_sampling([z[:, :-1] for z in Z], K)
                elif method == 'stealth':
                    q, _ = stealth_sampling.stealth_sampling([z[:, :-1] for z in Z], K, path='./', prefix='stealth', timeout=10.0)
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

                idx = np.random.choice(N, sum(K), p=np.concatenate(q), replace=False)
                res[alpha][method] = (q, idx)

            # save
            with open(p.joinpath('synthetic_%03d.pkl' % (seed,)), 'wb') as f:
                pickle.dump(res, f)


def evaluate(significance=0.05, seed=0):
    # outdir
    pout = pathlib.Path('./synthetic').joinpath('eval')
    pout.mkdir(parents=True, exist_ok=True)

    # load data
    p = pathlib.Path('./synthetic').joinpath('data')
    with open(p.joinpath('synthetic_%03d.pkl' % (seed,)), 'rb') as f:
        d = pickle.load(f)
    X, S, Y = d['te']
    Xref, Sref, _ = d['ref']
    Z = utils.split_to_four(X, S, Y)
    parity = utils.demographic_parity([z[:, -1] for z in Z])
    results = {'baseline':[parity, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]}

    # load sampling
    p = pathlib.Path('./synthetic').joinpath('sampling')
    with open(p.joinpath('synthetic_%03d.pkl' % (seed,)), 'rb') as f:
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

            test_res = ks_2samp(Xs[:, 0], Xref[:, 0])
            test_pos = ks_2samp(Xs[Ss>0.5, 0], Xref[Sref>0.5, 0])
            test_neg = ks_2samp(Xs[Ss<0.5, 0], Xref[Sref<0.5, 0])
            reject = test_res[1] < significance
            reject_pos = test_pos[1] < significance
            reject_neg = test_neg[1] < significance
            results[m] = [parity, test_res[1], reject, test_pos[1], reject_pos, test_neg[1], reject_neg]
            c = c + 1
    
        # save
        with open(pout.joinpath('synthetic_alpha%0.2f_%03d.pkl' % (alpha, seed)), 'wb') as f:
            pickle.dump(results, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['preprocess', 'sampling', 'evaluate'])
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--Nref', type=int, default=200)
    parser.add_argument('--Nsample', type=int, default=200)
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--spos', type=float, default=0.5)
    parser.add_argument('--ypos_coef', type=int, default=0.2)
    parser.add_argument('--n_parallel', type=int, default=-1)
    parser.add_argument('--eps', type=float, default=1e-2)
    parser.add_argument('--n_slice', type=int, default=10)
    parser.add_argument('--num_clusters', type=int, default=3)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--significance', type=float, default=0.05)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    args = parser.parse_args()

    if args.mode == 'preprocess':
        fn = lambda seed: preprocess_and_save(args.N, args.Nref, args.d, args.spos, args.ypos_coef, seed)
        joblib.Parallel(n_jobs=args.n_parallel)(joblib.delayed(fn)(seed) for seed in range(args.start, args.end))
    elif args.mode == 'sampling':
        for seed in range(args.start, args.end):
            sampling(args.N, args.Nsample, args.spos, args.n_slice, args.eps, args.num_clusters, args.max_depth, seed)
    elif args.mode == 'evaluate':
        fn = lambda seed: evaluate(args.significance, seed)
        n = (args.end - args.start) // 10
        for i in range(n):
            j = min(10 * (i + 1), args.end)
            joblib.Parallel(n_jobs=args.n_parallel)(joblib.delayed(fn)(seed) for seed in range(10*i, j))


