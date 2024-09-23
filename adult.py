import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib
import pathlib
import argparse

import utils
import stealth_sampling
import fast_stealth_sampling
import tree_stealth_sampling



def preprocess_and_save(classifier='LogReg', Ntr=10000, Nte=20000, seed=0):
    assert classifier in ['LogReg', 'Forest']

    # outdir
    p = pathlib.Path('./adult').joinpath('data').joinpath(classifier)
    p.mkdir(parents=True, exist_ok=True)
    
    # load data
    with open('adult.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # split data
    df_train, df_ref = train_test_split(df, test_size=1.0-Ntr/df.shape[0], random_state=seed)
    df_test, df_ref = train_test_split(df_ref, test_size=1.0-Nte/df_ref.shape[0], random_state=seed)

    # df to numpy array
    Xtr = df_train.drop(['Income', 'Sex: Male', 'Sex: Female'], axis=1).values
    Str = df_train['Sex: Male'].values
    Ytr = df_train['Income'].values
    Xte = df_test.drop(['Income', 'Sex: Male', 'Sex: Female'], axis=1).values
    Ste = df_test['Sex: Male'].values
    Yte = df_test['Income'].values
    Xref = df_ref.drop(['Income', 'Sex: Male', 'Sex: Female'], axis=1).values
    Sref = df_ref['Sex: Male'].values
    Yref = df_ref['Income'].values
    
    # normalize
    scaler = StandardScaler()
    scaler.fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    Xref = scaler.transform(Xref)
    
    # fit model
    if classifier == 'LogReg':
        model = LogisticRegressionCV(cv=3, random_state=seed)
    elif classifier == 'Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(np.c_[Xtr, Str], Ytr)
    Ttr = model.predict(np.c_[Xtr, Str])
    Tte = model.predict(np.c_[Xte, Ste])
    Tref = model.predict(np.c_[Xref, Sref])

    # save
    with open(p.joinpath('adult_%03d.pkl' % (seed,)), 'wb') as f:
        d = {'tr':[Xtr, Str, Ytr, Ttr], 'te':[Xte, Ste, Yte, Tte], 'ref':[Xref, Sref, Yref, Tref]}
        pickle.dump(d, f)


    
def sampling(classifier='LogReg', Nsample=2000, n_slice=10, eps=1e-1, num_clusters=7, max_depth=5, seed=0):
    assert classifier in ['LogReg', 'Forest']
    
    # load data
    p = pathlib.Path('./adult').joinpath('data').joinpath(classifier)
    with open(p.joinpath('adult_%03d.pkl' % (seed,)), 'rb') as f:
        d = pickle.load(f)
    Xte, Ste, _, Tte = d['te']
    Z = utils.split_to_four(Xte, Ste, Tte)

    # outdir
    p = pathlib.Path('./adult').joinpath('sampling').joinpath(classifier)
    p.mkdir(parents=True, exist_ok=True)


    # sampling
    alphas = np.linspace(0.1, 0.4, 4)
    methods = ['case-control', 'stealth', 'sliced', 'tree']
    res = {alpha:{method:[] for method in methods} for alpha in alphas}
    sampled_spos = np.mean(Ste)
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

            idx = np.random.choice(Xte.shape[0], sum(K), p=np.concatenate(q), replace=False)
            res[alpha][method] = (q, idx)

        # save
        with open(p.joinpath('adult_%03d.pkl' % (seed,)), 'wb') as f:
            pickle.dump(res, f)



def evaluate(classifier='LogReg', Nsample=2000, seed=0):
    assert classifier in ['LogReg', 'Forest']
    
    # outdir
    pout = pathlib.Path('./adult').joinpath('eval').joinpath(classifier)
    pout.mkdir(parents=True, exist_ok=True)

    # load data
    p = pathlib.Path('./adult').joinpath('data').joinpath(classifier)
    with open(p.joinpath('adult_%03d.pkl' % (seed,)), 'rb') as f:
        d = pickle.load(f)
    Xte, Ste, Tte, Yte = d['te']
    Xref, Sref, _, _ = d['ref']
    Z = utils.split_to_four(Xte, Ste, Tte)
    acc = 1.0 - np.mean(np.abs(Yte - Tte))
    parity = utils.demographic_parity([z[:, -1] for z in Z])
    
    # evaluate wassrstein distances
    np.random.seed(seed)
    idx = np.random.permutation(Xte.shape[0])[:Nsample]
    dx, dx_s1, dx_s0 = utils.compute_wasserstein_bootstrap(Xte[idx, :], Ste[idx], Xref, Sref, Nsample, prefix='%s_%03d_a' % (classifier, seed), num_sample=3, num_process=10, seed=seed)
    results = [acc, parity, dx, dx_s1, dx_s0]

    # save
    with open(pout.joinpath('adult_baseline_%03d.pkl' % (seed,)), 'wb') as f:
        pickle.dump(results, f)

    # load sampling
    p = pathlib.Path('./adult').joinpath('sampling').joinpath(classifier)
    with open(p.joinpath('adult_%03d.pkl' % (seed,)), 'rb') as f:
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
            dx, dx_s1, dx_s0 = utils.compute_wasserstein_bootstrap(Xs, Ss, Xref, Sref, Nsample, prefix='%s_%03d_b%02d' % (classifier, seed, c), num_sample=3, num_process=10, seed=seed)
            results[m] = [np.nan, parity, dx, dx_s1, dx_s0]
            c = c + 1
    
    # save
    with open(pout.joinpath('adult_alpha%0.2f_%03d.pkl' % (alpha, seed)), 'wb') as f:
        pickle.dump(results, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['preprocess', 'sampling', 'evaluate'])
    parser.add_argument('model', type=str, choices=['LogReg', 'Forest'], default='LogReg')
    parser.add_argument('--Ntr', type=int, default=10000)
    parser.add_argument('--Nte', type=int, default=20000)
    parser.add_argument('--Nsample', type=int, default=2000)
    parser.add_argument('--n_parallel', type=int, default=-1)
    parser.add_argument('--eps', type=float, default=1e-1)
    parser.add_argument('--n_slice', type=int, default=10)
    parser.add_argument('--num_clusters', type=int, default=7)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    args = parser.parse_args()

    if args.mode == 'preprocess':
        fn = lambda seed: preprocess_and_save(args.model, args.Ntr, args.Nte, seed)
        joblib.Parallel(n_jobs=args.n_parallel)(joblib.delayed(fn)(seed) for seed in range(args.start, args.end))
    elif args.mode == 'sampling':
        for seed in range(args.start, args.end):
            sampling(args.model,args.Nsample, args.n_slice, args.eps, args.num_clusters, args.max_depth, seed)
    elif args.mode == 'evaluate':
        for seed in range(args.start, args.end):
            evaluate(args.model, args.Nsample, seed)
        