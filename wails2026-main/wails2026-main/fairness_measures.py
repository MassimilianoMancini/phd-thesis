import pandas as pd
import numpy as np

def statistical_parity(df, sensible, classifier, split = None):
    if split is not None:
        df = df[df["SPLIT"] == split]

    return df.groupby(sensible)[classifier].mean()

def total_accuracy(df, sensible, classifier, success, split = None):
    if split is not None:
        df = df[df["SPLIT"] == split]

    return df.groupby(sensible).apply(lambda sub: (sub[classifier] == sub[success]).mean())

def pairwise_difference_matrix(series):
    values = series.astype(float).values
    diff = values[:, None] - values[None, :]
    return pd.DataFrame(diff, index=series.index, columns=series.index)

def ep_dm(matrix):
    values = matrix.values
    upper_indices = np.triu_indices_from(values, k=1)
    diffs = np.abs(values[upper_indices])
    if diffs.size == 0:
        return np.nan, np.nan
    ep = float(diffs.mean())
    dm = float(diffs.var())
    return ep, dm


def cosine_distance(series):
    values = series.astype(float).values
    k = values.size
    norm = np.sqrt((values ** 2).sum())
    if norm == 0:
        normalized = np.zeros_like(values)
    else:
        normalized = values / norm

    constant = 1.0 / np.sqrt(k)
    return float(0.5 * ((normalized - constant) ** 2).sum())

