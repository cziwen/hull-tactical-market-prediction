
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def zscore_by_time(df: pd.DataFrame, group_index: Optional[pd.Series] = None) -> pd.DataFrame:
    if df.index.nlevels == 1:
        return zscore_expanding(df)
    def _z(g):
        return (g - g.mean()) / g.std(ddof=0)
    return df.groupby(level=0).apply(_z)


def zscore_expanding(df: pd.DataFrame, min_periods: int = 63) -> pd.DataFrame:
    mean = df.expanding(min_periods=min_periods).mean()
    std = df.expanding(min_periods=min_periods).std(ddof=0)
    return (df - mean) / std


def rolling_ic(feature: pd.Series, target: pd.Series, window: int = 63) -> pd.Series:
    aligned = pd.concat({"f": feature, "y": target}, axis=1).dropna()
    return aligned["f"].rolling(window).corr(aligned["y"])


def icir(ic_series: pd.Series) -> float:
    ic = ic_series.dropna()
    if ic.empty:
        return np.nan
    daily = ic.mean() / (ic.std(ddof=0) + 1e-12)
    return float(daily * np.sqrt(252))


def corr_cluster(X: pd.DataFrame, threshold: float = 0.85) -> Tuple[Dict[str, List[str]], List[str]]:
    corr = X.corr().abs()
    remaining = set(X.columns)
    clusters = {}
    reps = []
    while remaining:
        hub = max(remaining, key=lambda c: corr.loc[c, list(remaining)].mean())
        cluster = [c for c in remaining if corr.loc[hub, c] >= threshold]
        clusters[hub] = cluster
        reps.append(hub)
        remaining -= set(cluster)
    return clusters, reps


def sign_flip_penalty(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return 0.0
    signs = np.sign(s.values)
    flips = np.sum(np.abs(np.diff(signs)) == 2)
    return flips / (len(signs) - 1)


@dataclass
class StabilityScore:
    feature: str
    mean_ic: float
    std_ic: float
    icir: float
    sign_flips: float
    score: float


def compute_stability_scores(X: pd.DataFrame, y: pd.Series, window: int = 63) -> List[StabilityScore]:
    results = []
    for col in X.columns:
        ic = rolling_ic(X[col], y, window=window)
        sflip = sign_flip_penalty(ic)
        score = ic.mean() - 0.5 * ic.std(ddof=0) - 0.25 * sflip
        results.append(StabilityScore(
            feature=col,
            mean_ic=float(ic.mean()),
            std_ic=float(ic.std(ddof=0)),
            icir=icir(ic),
            sign_flips=float(sflip),
            score=float(score)
        ))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def adf_stationarity(series: pd.Series) -> Optional[float]:
    if not HAS_STATSMODELS:
        return None
    s = series.dropna()
    if len(s) < 50:
        return None
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return None


def regime_labels(vol_series: pd.Series, pct: Tuple[float, float] = (0.33, 0.66)) -> pd.Series:
    q1, q2 = vol_series.quantile(pct[0]), vol_series.quantile(pct[1])
    def bucket(v):
        if v <= q1:
            return "low_vol"
        elif v <= q2:
            return "mid_vol"
        return "high_vol"
    return vol_series.apply(bucket)


def realized_vol(ret: pd.Series, span: int = 30) -> pd.Series:
    return ret.ewm(span=span, adjust=False).std() * np.sqrt(252)


def regime_ic_table(X: pd.DataFrame, y: pd.Series, regimes: pd.Series, window: int = 63) -> pd.DataFrame:
    rows = []
    df = pd.concat([X, y.rename("y"), regimes.rename("regime")], axis=1).dropna()
    for name in X.columns:
        for reg, g in df.groupby("regime"):
            ic = rolling_ic(g[name], g["y"], window=window)
            rows.append({"feature": name, "regime": reg, "mean_ic": ic.mean(), "icir": icir(ic)})
    return pd.DataFrame(rows).pivot(index="feature", columns="regime", values="mean_ic")


def time_series_cv_splits(n_splits: int, n_samples: int, test_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    step = (n_samples - test_size) // max(n_splits, 1)
    for i in range(n_splits):
        train_end = (i + 1) * step
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, min(train_end + test_size, n_samples))
        if len(test_idx) == 0:
            break
        splits.append((train_idx, test_idx))
    return splits


def permutation_importance_on_timeslice(model, X_val: pd.DataFrame, y_val: pd.Series, scorer: Callable[[np.ndarray, np.ndarray], float], n_repeats: int = 10, random_state: int = 42) -> Dict[str, float]:
    base = scorer(y_val.values, model.predict(X_val).reshape(-1))
    drops = {}
    rng = np.random.default_rng(random_state)
    for col in X_val.columns:
        scores = []
        for _ in range(n_repeats):
            Xp = X_val.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            pred = model.predict(Xp).reshape(-1)
            scores.append(base - scorer(y_val.values, pred))
        drops[col] = float(np.mean(scores))
    return drops


def sharpe_like(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pnl = y_true * y_pred
    mu = pnl.mean()
    sd = pnl.std(ddof=0) + 1e-12
    return float(mu / sd)


def select_features_stable(
    X: pd.DataFrame,
    y: pd.Series,
    top_k_per_cluster: int = 1,
    corr_threshold: float = 0.85,
    rolling_window: int = 63,
    max_features: int = 40
) -> List[str]:
    scores = compute_stability_scores(X, y, window=rolling_window)
    rank_order = [s.feature for s in scores]
    clusters, reps = corr_cluster(X[rank_order], threshold=corr_threshold)
    rep_order = sorted(reps, key=lambda f: rank_order.index(f))
    selected = rep_order[:max_features]
    return selected


def lasso_stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_boot: int = 50,
    sample_frac: float = 0.7,
    alpha_grid: Optional[np.ndarray] = None,
    random_state: int = 42
) -> pd.Series:
    from sklearn.linear_model import LassoCV
    rng = np.random.default_rng(random_state)
    if alpha_grid is None:
        alpha_grid = np.logspace(-4, 0, 30)
    counts = pd.Series(0, index=X.columns, dtype=float)
    for b in range(n_boot):
        idx = rng.choice(np.arange(len(X)), size=int(len(X) * sample_frac), replace=False)
        Xb, yb = X.iloc[idx], y.iloc[idx]
        model = LassoCV(alphas=alpha_grid, cv=3, random_state=42).fit(Xb, yb)
        sel = np.abs(model.coef_) > 1e-8
        counts.loc[X.columns[sel]] += 1
    return counts / n_boot


def prepare_model_ready(
    features: pd.DataFrame,
    target: pd.Series,
    is_scored: Optional[pd.Series] = None,
    mode: str = "lean",
    max_features: int = 40
) -> Tuple[pd.DataFrame, List[str]]:
    df = features.copy()
    na_ratio = df.isna().mean()
    keep = na_ratio[na_ratio <= 0.3].index.tolist()
    df = df[keep].copy()
    df = df.fillna(method="ffill").fillna(method="bfill")
    aligned = pd.concat([df, target.rename("target")], axis=1).dropna()
    X = aligned.drop(columns=["target"])
    y = aligned["target"]
    selected = select_features_stable(X, y, max_features=max_features)
    if mode == "full":
        scores = compute_stability_scores(X, y)
        order = [s.feature for s in scores]
        extra = [c for c in order if c not in selected][:max_features]
        selected = selected + extra
    df_out = aligned[selected + ["target"]].copy()
    return df_out, selected
