"""The article's two reader-facing tools, plus a few helpers.

1. correlation(): "does one thing go along with another?"
   Spearman rank correlation, a bootstrap 95% confidence interval, and a
   shuffle-test p-value.
2. shuffle_test() / paired_shuffle_test(): "is this real, or just luck?"
   Shuffle the labels many times and see how often chance beats reality.

Everything uses a fixed seed so results are identical on every run.
"""
import numpy as np
from scipy.stats import rankdata, spearmanr

SEED = 42
N_SHUFFLES = 10_000


def _rowwise_pearson(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=1) / denom


def _p_value(null, observed, alternative):
    """Shuffle-test p-value with the +1 correction (never exactly zero)."""
    if alternative == "two-sided":
        hits = np.sum(np.abs(null) >= abs(observed))
    elif alternative == "greater":
        hits = np.sum(null >= observed)
    else:
        hits = np.sum(null <= observed)
    return (hits + 1) / (len(null) + 1)


def verdict(rho, p):
    """Plain-English label readers see instead of numbers."""
    if p >= 0.05 or abs(rho) < 0.1:
        return "no clear link"
    size = "weak" if abs(rho) < 0.3 else "moderate" if abs(rho) < 0.5 else "strong"
    return f"{size} {'positive' if rho > 0 else 'negative'} link"


def correlation(x, y, n=N_SHUFFLES, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    assert len(x) == len(y) and not (np.isnan(x).any() or np.isnan(y).any())
    rng = np.random.default_rng(seed)
    rho = spearmanr(x, y).statistic
    rx, ry = rankdata(x), rankdata(y)

    # Shuffle test: break the pairing, see how big rho gets by chance
    shuffled = np.array([rng.permutation(ry) for _ in range(n)])
    null = _rowwise_pearson(np.broadcast_to(rx, shuffled.shape), shuffled)
    p = _p_value(null, rho, "two-sided")

    # Bootstrap CI: resample Pokémon with replacement, re-rank, recompute
    idx = rng.integers(0, len(x), size=(n, len(x)))
    boot = _rowwise_pearson(rankdata(x[idx], axis=1), rankdata(y[idx], axis=1))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])

    return {"rho": round(float(rho), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "p": float(p), "n": int(len(x)), "verdict": verdict(rho, p),
            "scipy_p": float(spearmanr(x, y).pvalue)}


def shuffle_test(stat_fn, labels, n=N_SHUFFLES, seed=SEED, alternative="greater"):
    """stat_fn(labels) -> number. Shuffles `labels` n times to build the
    'just luck' distribution and compares the real value against it."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    observed = stat_fn(labels)
    null = np.array([stat_fn(rng.permutation(labels)) for _ in range(n)])
    return {"observed": float(observed), "chance_median": float(np.median(null)),
            "chance_95th": float(np.percentile(null, 95)),
            "p": float(_p_value(null, observed, alternative)), "n_shuffles": n}


def paired_shuffle_test(diffs, n=N_SHUFFLES, seed=SEED):
    """For before/after pairs: if evolving changed nothing, each gain is
    equally likely to be a loss, so randomly flip signs and compare means."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, float)
    observed = diffs.mean()
    signs = rng.choice([-1.0, 1.0], size=(n, len(diffs)))
    null = (signs * np.abs(diffs)).mean(axis=1)
    return {"observed_mean": float(observed), "p": float(_p_value(null, observed, "two-sided")),
            "n_pairs": int(len(diffs)), "n_shuffles": n}


def benjamini_hochberg(pvals):
    """Adjust p-values for running many tests at once (appendix only)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out.tolist()
