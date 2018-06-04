import itertools
import numpy as np
import xarray as xr
from .estimators import transfer_entropy


def estimate_timescales(X, Y, lag_list, window_list, sample_size=5000):
    out = xr.DataArray(np.zeros((len(lag_list), len(window_list))),
                       coords={'lag': lag_list, 'window': window_list},
                       dims=('lag', 'window'))
    for l, w in itertools.product(lag_list, window_list):
        ss = np.min([sample_size, (len(X)-l-w)//2])
        max_start = len(X) - l - w - ss
        si = np.random.randint(0, max_start)
        Xs, Ys = X[si:si+ss], Y[si:si+ss]
        # good_data = reduce(np.intersect1d, [good_inds(v) for v in [Xs, Ys]])
        # Xs, Ys = X[good_data], Y[good_data]
        out.loc[{'lag': l, 'window': w}] = transfer_entropy(Xs, Ys, l, 1, w, 1)
    return out


def estimator_stats(estimator: callable, data: dict,
                    params: dict, nruns: int, sample_size: int) -> dict:
    results = []
    for _ in range(nruns):
        X = list(data.values())[0]
        l, w = params.get('l', 0), params.get('w', 0)
        ss = np.min([sample_size, (len(X)-l-w)//2])
        max_start = len(X) - l - w - ss
        si = np.random.randint(0, max_start)
        data2 = {k: v[si:si+ss] for k, v in data.items()}
        results.append(estimator(**data2, **params))
    statistics = {'mean': np.mean(results),
                  'median': np.median(results),
                  'variance': np.var(results),
                  'max': np.max(results),
                  'min': np.min(results),
                  'results': results}
    return statistics
