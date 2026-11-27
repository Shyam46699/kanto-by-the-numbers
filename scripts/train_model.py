"""AI finale: can a model guess a Pokémon's strength without seeing battle stats?

Clues:   size (height, weight), colour, catch rate, evolution stage,
         fully evolved, legendary/mythical
Target:  strength = total of the six battle stats
Fairness: leave-one-out, so every Pokémon is guessed by a model that never saw it.

Reads:   data/processed/gen1.csv
Writes:  results/model_predictions.csv, results/model_importance.csv,
         and a "model" section in results/results.json

Usage:  ~/.venvs/pokemon-ml/bin/python scripts/train_model.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold, cross_val_predict, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "gen1.csv"
OUT = ROOT / "results"
SEED = 42


def features(df):
    X = pd.DataFrame({
        "log_height": np.log(df["height_m"]),
        "log_weight": np.log(df["weight_kg"]),
        "capture_rate": df["capture_rate"],
        "evo_stage": df["evo_stage"],
        "fully_evolved": (df["fully_evolved"].astype(str) == "True").astype(int),
        "legendary": ((df["legendary"].astype(str) == "True")
                      | (df["mythical"].astype(str) == "True")).astype(int),
    })
    X = X.join(pd.get_dummies(df["color"], prefix="colour").astype(int))
    groups = {
        "Size": ["log_height", "log_weight"],
        "Colour": [c for c in X if c.startswith("colour_")],
        "Rarity (catch rate)": ["capture_rate"],
        "Evolution stage": ["evo_stage", "fully_evolved"],
        "Legendary": ["legendary"],
    }
    assert not X.isna().any().any()
    return X, groups


def models():
    return {
        "Guess the average (baseline)": DummyRegressor(strategy="mean"),
        "Straight-line model (ridge)": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Random forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    }


def group_importance(model, X, y, groups, n_repeats=10):
    """Shuffle one clue group at a time on held-out Pokémon and measure how
    much worse the guesses get (in strength points)."""
    rng = np.random.default_rng(SEED)
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    drops = {g: [] for g in groups}
    for train, test in cv.split(X):
        m = model.fit(X.iloc[train], y.iloc[train])
        Xt, yt = X.iloc[test], y.iloc[test]
        base = mean_absolute_error(yt, m.predict(Xt))
        for g, cols in groups.items():
            Xs = Xt.copy()
            perm = rng.permutation(len(Xs))
            Xs[cols] = Xs[cols].to_numpy()[perm]  # shuffle the group's columns together
            drops[g].append(mean_absolute_error(yt, m.predict(Xs)) - base)
    return pd.DataFrame({
        "clue": list(drops),
        "extra_error_points": [round(float(np.mean(v)), 1) for v in drops.values()],
        "sd": [round(float(np.std(v)), 1) for v in drops.values()],
    }).sort_values("extra_error_points", ascending=False)


def model_permutation_test(X, y, n_permutations=1000):
    """Is the forest's accuracy better than chance? Shuffle the strength values, retrain and
    re-score (5-fold cross-validation) many times; p = share of shuffles that do as well as
    the real data. Uses 5-fold rather than leave-one-out to keep thousands of refits feasible."""
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=1)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    score, perm_scores, p = permutation_test_score(
        rf, X, y, cv=cv, n_permutations=n_permutations, scoring="neg_mean_absolute_error",
        random_state=SEED, n_jobs=-1)
    return {"method": f"permutation_test_score, {n_permutations} shuffles of strength, 5-fold CV",
            "real_mae": round(float(-score), 1),
            "shuffled_mae_best": round(float(-perm_scores.max()), 1),
            "shuffled_mae_median": round(float(-np.median(perm_scores)), 1),
            "n_permutations": n_permutations, "p": float(p)}


def main():
    df = pd.read_csv(DATA, keep_default_na=False)
    assert len(df) == 151
    X, groups = features(df)
    y = df["total"]

    scores, preds = {}, {}
    for name, model in models().items():
        p = cross_val_predict(model, X, y, cv=LeaveOneOut(), n_jobs=1)
        assert len(p) == 151
        preds[name] = p
        scores[name] = {"typical_error_points": round(float(mean_absolute_error(y, p)), 1),
                        "r2": round(float(r2_score(y, p)), 3)}
        print(f"{name:32s} MAE {scores[name]['typical_error_points']:6.1f}   R² {scores[name]['r2']:.3f}")

    contenders = [n for n in scores if "baseline" not in n]
    best = min(contenders, key=lambda n: scores[n]["typical_error_points"])
    baseline = scores["Guess the average (baseline)"]["typical_error_points"]
    beats_baseline = scores[best]["typical_error_points"] < baseline

    out = df[["num", "name", "total"]].copy()
    out["ai_guess"] = preds[best].round().astype(int)
    out["miss"] = out["total"] - out["ai_guess"]  # + = stronger than it looks
    out.to_csv(OUT / "model_predictions.csv", index=False)

    importance = group_importance(models()[best], X, y, groups)
    importance.to_csv(OUT / "model_importance.csv", index=False)

    ranked = out.sort_values("miss")
    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    results["model"] = {
        "clues": list(groups), "target": "strength (total of six battle stats)",
        "testing": "leave-one-out: each Pokémon guessed by a model that never saw it",
        "scores": scores, "best_model": best, "beats_baseline": beats_baseline,
        "error_reduction_vs_baseline_pct": round((1 - scores[best]["typical_error_points"] / baseline) * 100, 1),
        "importance": importance.to_dict("records"),
        "permutation_test": model_permutation_test(X, y),
        "stronger_than_it_looks": ranked.iloc[::-1].head(5)[["name", "total", "ai_guess", "miss"]].to_dict("records"),
        "weaker_than_it_looks": ranked.head(5)[["name", "total", "ai_guess", "miss"]].to_dict("records"),
    }
    results_path.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False))
    print(f"Best: {best}. Beats baseline: {beats_baseline}")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    import sys
    if "--permutation-test-only" in sys.argv:   # add the test to existing results without refitting everything
        df = pd.read_csv(DATA, keep_default_na=False); X, _ = features(df)
        path = OUT / "results.json"; results = json.loads(path.read_text())
        results["model"]["permutation_test"] = model_permutation_test(X, df["total"])
        path.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False))
        print(results["model"]["permutation_test"])
    else:
        main()
