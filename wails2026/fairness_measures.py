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

def cosine_distance_equity(series, reference):
    """
    Equity-based counterpart of cosine_distance(). Instead of comparing the
    observed group-wise vector e to the fixed equality reference (1/sqrt(k), ...),
    this compares it to a custom reference vector eq (Section 2.4 of the paper),
    e.g. eq_i = M_i / N_i for a chosen target M_i per group.

    Both e and eq are L2-normalized (Eq. 4) before comparison, so `reference`
    does NOT need to be pre-normalized or unit-length: any vector of positive
    target rates, one per group, is accepted.

    Uses the same identity as cosine_distance(): for unit vectors e', eq',
    0.5 * ||e' - eq'||^2 = 1 - cos(theta), i.e. Eq. (7) divided by 2.

    Parameters
    ----------
    series : pandas.Series
        Observed group-wise measure e_i (output of statistical_parity or
        total_accuracy), indexed by group.
    reference : pandas.Series or array-like
        Custom equity reference eq_i. If a pandas.Series, it is reindexed to
        match `series`'s group order (so index labels must match). If a plain
        array-like, it must already be in the same group order as `series`.

    Returns
    -------
    float
        1 - cos(theta) between the normalized observed vector and the
        normalized equity reference vector. NaN if either vector is all zeros.
    """
    values = series.astype(float).values

    if isinstance(reference, pd.Series):
        reference = reference.reindex(series.index)
        if reference.isna().any():
            missing = reference[reference.isna()].index.tolist()
            raise ValueError(f"reference is missing values for groups: {missing}")
    ref_values = np.asarray(reference, dtype=float)

    if ref_values.shape != values.shape:
        raise ValueError("series and reference must have the same length (one value per group)")

    v_norm = np.sqrt((values ** 2).sum())
    r_norm = np.sqrt((ref_values ** 2).sum())
    if v_norm == 0 or r_norm == 0:
        return np.nan

    v_normalized = values / v_norm
    r_normalized = ref_values / r_norm

    return float(0.5 * ((v_normalized - r_normalized) ** 2).sum())

def group_cardinality(df, sensible):
    """
    Compute the number of samples in each group defined by the `sensible` column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    sensible : str
        The name of the column in `df` that defines the groups.

    Returns
    -------
    pandas.Series
        A Series indexed by group labels, containing the count of samples in each group.
    """
    return df.groupby(sensible).size()

def total_positives_sp(df, sensible, classifier):
    """
    Compute the total number of positive outcomes in each group defined by the `sensible` column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    sensible : str
        The name of the column in `df` that defines the groups.
    classifier : str
        The name of the column in `df` that contains the classifier predictions.

    Returns
    -------
    pandas.Series
        A Series indexed by group labels, containing the count of positive outcomes in each group.
    """
    return df.groupby(sensible)[classifier].sum()

def total_positives_ta(df, sensible, classifier, success):
    """
    Compute the total number of positive outcomes in each group defined by the `sensible` column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    sensible : str
        The name of the column in `df` that defines the groups.
    classifier : str
        The name of the column in `df` that contains the classifier predictions.
    success : str
        The name of the column in `df` that indicates positive outcomes (e.g., 1 for success).

    Returns
    -------
    pandas.Series
        A Series indexed by group labels, containing the count of positive outcomes in each group.
    """
    return df.groupby(sensible).apply(lambda sub: (sub[classifier] == sub[success]).sum())

def total_positives_ref(df, sensible, ref):
    """
    Compute the total number of positive outcomes in each group defined by the `sensible` column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    sensible : str
        The name of the column in `df` that defines the groups.
    ref : pandas.Series
        A Series indexed by group labels, containing the reference values for each group.

    Returns
    -------
    float
        The sum of the reference value multiplied by the cardinality of each group.
    """
    cardinalities = group_cardinality(df, sensible)
    return sum(ref[group] * cardinalities[group] for group in cardinalities.index)

