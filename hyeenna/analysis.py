import os
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as st
from joblib import Parallel, delayed
from .estimators import entropy
from .estimators import transfer_entropy as te
from .estimators import conditional_transfer_entropy as cte


def estimate_timescales(X: np.ndarray, Y: np.ndarray,
                        lag_list: list, window_list: list,
                        sample_size: int=5000) -> pd.DataFrame:
    """
    Compute the transfer entropy (TE) over a range of lag counts and
    window sizes.

    Parameters
    ----------
    X: np.array
        Source data
    Y: np.array
        Target data
    lag_list: list
        A list enumerating the lag counts to compute TE with
    window_list: list
        A list enumerating the window widths to compute TE with
    sample_size: int
        Number of samples to use when computing TE

    Returns
    -------
    out: pd.DataFrame
        A dataframe containing the computed transfer entropies
        for every combination of lag and window given in the
        input parameters
    """
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
    """
    The kernel function of the `estimator_stats` function. Runs a single
    instance of the estimator.
    """
    # Setting seed to None keeps joblib from blowing up
    np.random.seed(None)
    X = list(data.values())[0]
    l, w = params.get('l', 1), params.get('omega', 1)
    ss = np.min([sample_size, (len(X)-l-w)])
    max_start = len(X) - l - w - ss
    if max_start < 1:
        raise ValueError("Maximum start index is negative!"
                         + os.linesep + "Computed value {}".format(max_start)
                         + os.linesep + "Components leading to this error:"
                         + os.linesep + " Data length: {}".format(len(X))
                         + os.linesep + " Lag size: {}".format(l)
                         + os.linesep + " Window size: {}".format(w)
                         + os.linesep + " Sample size: {}".format(ss)
                         )
    si = np.random.randint(0, max_start)
    data2 = {k: v[:][si:si+ss] for k, v in data.items()}
    return estimator(**data2, **params)


def estimator_stats(estimator: callable, data: dict, params: dict,
                    nruns: int=10, sample_size: int=3000) -> dict:
    """
    Compute some statistics about a given estimator.

    Parameters
    ----------
    estimator: callable
        The estimator to compute statistics on. Suggested
        to be from the HYEENNA library.
    data: dict
        Input data to feed into the estimator
    params: dict
        Parameters to feed into the estimator
    nruns: int (default: 10)
        Number of times to run the estimator.
    sample_size: int (default 3000)
        Size of sample to draw from `data` to feed into the estimator

    Returns
    -------
    stats: dict
        A dictionary containing sample statistics along with the actual
        results from each run of the estimator.
    """
    results = Parallel(n_jobs=nruns)(delayed(_run_one_estimator_stats)(
        estimator, data, params, sample_size) for i in range(nruns))
    stats = {'mean': np.mean(results),
             'median': np.median(results),
             'variance': np.var(results, ddof=1),
             'max': np.max(results),
             'min': np.min(results),
             'results': results}
    return stats


def _run_one_shuffle_test(estimator, data, params, sample_size):
    """
    The kernel function of the `shuffle_test` function. Runs a single
    instance of the estimator on a shuffled random surrogate.
    """
    np.random.seed(None)
    X = list(data.values())[0]
    l, w = params.get('l', 1), params.get('omega', 1)
    ss = np.min([sample_size, (len(X)-l-w)//2])
    max_start = len(X) - l - w - ss
    if max_start < 1:
        raise ValueError("Maximum start index is negative!"
                         + os.linesep + "Computed value {}".format(max_start)
                         + os.linesep + "Components leading to this error:"
                         + os.linesep + " Data length: {}".format(len(X))
                         + os.linesep + " Lag size: {}".format(l)
                         + os.linesep + " Window size: {}".format(w)
                         + os.linesep + " Sample size: {}".format(ss))
    si = np.random.randint(0, max_start)
    data2 = {key: val[:][si:si+ss].copy() for key, val in data.items()}
    for key, val in data2.items():
        np.random.shuffle(val)
    return estimator(**data2, **params)


def shuffle_test(estimator: callable, data: dict,
                 params: dict, confidence: float=0.99,
                 nruns: int=10, sample_size: int=3000) -> dict:
    """
    Compute a one tailed Z test against a sample of shuffled surrogates.

    Parameters
    ----------
    estimator: callable
        The estimator to compute statistics on. Suggested
        to be from the HYEENNA library.
    data: dict
        Input data to feed into the estimator
    params: dict
        Parameters to feed into the estimator
    confidence: float (default: 0.99)
        Confidence level to conduct the test at.
    nruns: int (default: 10)
        Number of times to run the estimator.
    sample_size: int (default: 3000)
        Size of sample to draw from `data` to feed into the estimator

    Returns
    -------
    stats: dict
        A dictionary with statistics from the standard `estimator_stats`
        function along with statistics computed on the shuffled surrogates.
        Most importantly are the 'test_value' and 'significant' keys, which
        are the value to perform the test on, along with whether the test
        result was significantly significant at the given confidence level.
    """
    stats = estimator_stats(estimator, data, params, nruns, sample_size)
    stats['test_value'] = np.random.choice(stats['results'])

    shuffled_te = Parallel(n_jobs=nruns)(delayed(_run_one_shuffle_test)(
        estimator, data, params, sample_size) for i in range(nruns))
    stats['shuffled_results'] = shuffled_te
    stats['ci'] = [np.percentile(shuffled_te, 1),
                   np.percentile(shuffled_te, 99)]
    stats['shuffled_median'] = np.median(shuffled_te)
    stats['shuffled_mean'] = np.mean(shuffled_te)
    stats['shuffled_variance'] = np.var(shuffled_te, ddof=1)
    stats['shuffled_thresh'] = (stats['shuffled_mean']
                                + st.norm.ppf(confidence)
                                * stats['shuffled_variance'])
    stats['significant'] = stats['test_value'] > stats['shuffled_thresh']
    return stats


def estimate_info_transfer_network(varlist: list, names: list,
                                   tau: int=1, omega: int=1, nu: int=1,
                                   k: int=1, l: int=1, m: int=1,
                                   condition: bool=True, nruns: int=10,
                                   sample_size: int=3000) -> pd.DataFrame:
    """
    Compute the pairwise transfer entropy for a list of given variables,
    resulting in an information transfer network.

    Parameters
    ----------
    varlist: list
        List of given variable data
    names: list
        List of names corresponding to the data given in `varlist`
    tau: int (default=1)
        Lag value for source variables
    omega: int (default=1)
        Lag value for conditioning target variable history
    nu: int (default=1)
        Lag value for conditioning source variable histories
    k: int (default=1)
        Window length for source variables (applied to the
        same variable as the `tau` parameter)
    l: int (default=1)
        Window length for target variable histories (applied
        to the same variable as the `omega` parameter)
    m: int (default=1)
        Window length for source conditioning variables (applied
        to the same variable as the `nu` parameter)
    condition: bool (default=False)
        Whether to condition on all variables, or just the target
        variable history.
    nruns: int (default=10)
        Number of samples to compute for each connection. The median
        value is reported.
    sample_size: int (default=3000)
        Size of samples to take during estimation of transfer entropy.

    Returns
    -------
    df: pd.DataFrame
        Dataframe representing the information transfer network. Both rows
        and columns are populated with the given `names`.
    """
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
        scores.append(res['mean'] * res['significant'])
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
