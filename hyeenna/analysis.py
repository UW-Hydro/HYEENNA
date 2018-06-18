import itertools
import numpy as np
import pandas as pd
import xarray as xr
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


def estimator_stats(estimator: callable, data: dict, params: dict,
                    nruns: int=10, sample_size: int=3000) -> dict:
    results = []
    for _ in range(nruns):
        X = list(data.values())[0]
        l, w = params.get('l', 1), params.get('omega', 1)
        ss = np.min([sample_size, (len(X)-l-w)//2])
        max_start = len(X) - l - w - ss
        si = np.random.randint(0, max_start)
        data2 = {k: v[:][si:si+ss] for k, v in data.items()}
        results.append(estimator(**data2, **params))
    statistics = {'mean': np.mean(results),
                  'median': np.median(results),
                  'variance': np.var(results),
                  'max': np.max(results),
                  'min': np.min(results),
                  'results': results}
    return statistics


def shuffle_test(estimator: callable, data: dict,
                 params: dict, nruns: int=10,
                 sample_size: int=3000, ci: float=0.95) -> dict:
    stats = estimator_stats(estimator, data, params, nruns, sample_size)
    shuffled_te = []
    for i in range(nruns):
        X = list(data.values())[0]
        l, o = params.get('l', 1), params.get('omega', 1)
        ss = np.min([sample_size, (len(X)-l-o)//2])
        max_start = len(X) - l - o - ss
        si = np.random.randint(0, max_start)
        data2 = {key: val[:][si:si+ss] for key, val in data.items()}
        for key, val in data2.items():
            np.random.shuffle(val)
        shuffled_te.append(estimator(**data2, **params))

    stats['shuffled_results'] = shuffled_te
    stats['ci'] = [np.percentile(shuffled_te, 5),
                   np.percentile(shuffled_te, 95)]
    stats['shuffled_median'] = np.median(shuffled_te)
    stats['significant'] = ((stats['median'] < stats['ci'][0])
                            or (stats['median'] > stats['ci'][1]))
    return stats


def estimate_network(varlist: list, names: list, out_file: str,
                     tau: int=1, omega: int=1, k: int=1, l: int=1,
                     condition: bool=True, test_significance: bool=True,
                     nruns: int=10, sample_size: int=3000) -> pd.DataFrame:
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
    params = {'tau': tau, 'omega': omega, 'k': k, 'l': l}
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

        if test_significance:
            res = shuffle_test(**args)
            if res['significant']:
                scores.append(res['median'])
            else:
                scores.append(0.0)
        else:
            res = estimator_stats(**args)
            scores.append(res['median'])
    # Reformat into a nice dataframe, save it, and return
    df = pd.DataFrame(columns=names, index=names)
    for link, score in zip(permutations, scores):
        if score < 1e-4:
            score = 0
        df.loc[link[0], link[1]] = score
    for name, var in zip(names, varlist):
        e_tot = entropy(var)
        tot_exp = df.loc[name, :].sum()
        df[name][name] = e_tot - tot_exp
    df.to_csv('./data/{}.csv'.format(out_file))
    return df
