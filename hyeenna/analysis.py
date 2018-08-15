import itertools
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed, dump, load
from .estimators import entropy
from .estimators import transfer_entropy as te
from .estimators import conditional_transfer_entropy as cte


def estimate_timescales(X, Y, lag_list, window_list, sample_size=5000):
    out = xr.DataArray(np.zeros((len(lag_list), len(window_list))),
                       coords={'lag': lag_list, 'window': window_list},
                       dims=('lag', 'window'))
    for l, w in itertools.product(lag_list, window_list):
        ss = np.min([sample_size, (len(X)-l-w)//2])
        max_start = len(X) - l - w - ss
        si = np.random.randint(0, max_start)
        Xs, Ys = X[si:si+ss], Y[si:si+ss]
        out.loc[{'lag': l, 'window': w}] = te(Xs, Ys, l, 1, w, 1)
    return out


def _run_one_estimator_stats(estimator, data, params, sample_size):
    np.random.seed(None)
    X = list(data.values())[0]
    l, w = params.get('l', 1), params.get('omega', 1)
    ss = np.min([sample_size, (len(X)-l-w)//2])
    max_start = len(X) - l - w - ss
    si = np.random.randint(0, max_start)
    data2 = {k: v[:][si:si+ss] for k, v in data.items()}
    return estimator(**data2, **params)


def estimator_stats(estimator: callable, data: dict, params: dict,
                    nruns: int=10, sample_size: int=3000) -> dict:
    results = Parallel(n_jobs=nruns)(delayed(_run_one_estimator_stats)(
        estimator, data, params, sample_size) for i in range(nruns))
    statistics = {'mean': np.mean(results),
                  'median': np.median(results),
                  'variance': np.var(results, ddof=1),
                  'max': np.max(results),
                  'min': np.min(results),
                  'results': results}
    return statistics


def _run_one_shuffle_test(estimator, data, params, sample_size):
    np.random.seed(None)
    X = list(data.values())[0]
    l, o = params.get('l', 1), params.get('omega', 1)
    ss = np.min([sample_size, (len(X)-l-o)//2])
    max_start = len(X) - l - o - ss
    si = np.random.randint(0, max_start)
    data2 = {key: val[:][si:si+ss].copy() for key, val in data.items()}
    for key, val in data2.items():
        np.random.shuffle(val)
    return estimator(**data2, **params)


def shuffle_test(estimator: callable, data: dict,
                 params: dict, nruns: int=10,
                 sample_size: int=3000) -> dict:
    stats = estimator_stats(estimator, data, params, nruns, sample_size)

    shuffled_te = Parallel(n_jobs=nruns)(delayed(_run_one_shuffle_test)(
        estimator, data, params, sample_size) for i in range(nruns))
    stats['shuffled_results'] = shuffled_te
    stats['ci'] = [np.percentile(shuffled_te, 1),
                   np.percentile(shuffled_te, 99)]
    stats['shuffled_median'] = np.median(shuffled_te)
    stats['shuffled_mean'] = np.mean(shuffled_te)
    stats['shuffled_variance'] = np.var(shuffled_te, ddof=1)
    c = 2.36
    stats['shuffled_thresh'] = (
            stats['shuffled_mean'] + c * stats['shuffled_variance'])
    stats['significant'] = stats['median'] > stats['shuffled_thresh']
    return stats


def estimate_network(varlist: list, names: list,
                     tau: int=1, omega: int=1, nu: int=1,
                     k: int=1, l: int=1, m: int=1,
                     condition: bool=True, nruns: int=10,
                     sample_size: int=3000) -> pd.DataFrame:
    # Calculate all needed variable combinations
    mapping = {n: d for n, d in zip(names, varlist)}
    permutations = [list(l) for l in list(itertools.permutations(names, 2))]
    for combo in permutations:
        n = [n for n in names if n not in combo]
        [combo.append(nn) for nn in n]
    # Subsample data and put it together with combination list
    analysis_sets = []
    for combo in permutations:
        analysis_sets.append([mapping[c] for c in combo])
    # Compute scores
    scores = []
    params = {'tau': tau, 'omega': omega, 'nu': nu, 'k': k, 'l': l, 'm': m}
    for c, s in zip(permutations, analysis_sets):
        if condition:
            X = np.array(s[0]).reshape(-1, 1)
            Y = np.array(s[1]).reshape(-1, 1)
            Z = np.array(s[2:]).T
            args = {'estimator': cte,
                    'data': {'X': X, 'Y': Y, 'Z': Z},
                    'params': params, 'nruns': nruns,
                    'sample_size': sample_size}
        else:
            X, Y = np.array(s[0]).reshape(-1, 1), np.array(s[1]).reshape(-1, 1)
            args = {'estimator': te,
                    'data': {'X': X, 'Y': Y},
                    'params': params, 'nruns': nruns,
                    'sample_size': sample_size}

        res = shuffle_test(**args)
        scores.append(res['median'] * res['significant'])
    # Reformat into a dataframe
    df = pd.DataFrame(columns=names, index=names)
    for link, score in zip(permutations, scores):
        if score < 1e-4:
            score = 0
        df.loc[link[0], link[1]] = score
    for name, var in zip(names, varlist):
        e_tot = entropy(var)
        tot_exp = df.loc[name, :].sum()
        df[name][name] = e_tot - tot_exp
    return df
