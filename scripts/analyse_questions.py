"""Answer the article's five questions about the original 151 Pokémon.

Reads:   data/processed/gen1.csv
Writes:  results/results.json         every number the article quotes
         results/q1_rarity.csv ... q5_evolution.csv   tidy data for charts

"Strength" = total of the six battle stats.

Usage:  ~/.venvs/pokemon-ml/bin/python scripts/analyse_questions.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_tools import benjamini_hochberg, correlation, paired_shuffle_test, shuffle_test  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "gen1.csv"
OUT = ROOT / "results"
STATS = ["hp", "attack", "defense", "sp_atk", "sp_def", "speed"]


def load():
    df = pd.read_csv(DATA, keep_default_na=False, na_values={"pct_female": [""]})
    for col in ["legendary", "mythical", "genderless", "fully_evolved"]:
        df[col] = df[col].astype(str) == "True"
    assert len(df) == 151 and df["num"].is_unique
    return df


def names(frame, n=5):
    return frame["name"].head(n).tolist()


# ---------------------------------------------------------------- Q1
def q1_rarity(df):
    d = df.copy()
    d["rarity"] = 255 - d["capture_rate"]  # 0 = easiest to catch, 252 = hardest
    res = {"overall": correlation(d["rarity"], d["total"])}

    # "Best value": much stronger than its rarity suggests (and the reverse)
    d["value_gap"] = d["total"].rank(pct=True) - d["rarity"].rank(pct=True)
    d = d.sort_values("value_gap", ascending=False)
    res["best_value"] = names(d)
    res["worst_value"] = names(d.iloc[::-1])
    res["legendaries"] = d.loc[d["legendary"] | d["mythical"],
                               ["name", "capture_rate", "total"]].to_dict("records")
    res["strength_rank_of_legendaries"] = {
        n: int(r) for n, r in zip(df["name"], df["total"].rank(ascending=False, method="min"))
        if n in {x["name"] for x in res["legendaries"]}}

    d["highlight"] = np.select(
        [d["legendary"], d["mythical"], d["name"].isin(res["best_value"]),
         d["name"].isin(res["worst_value"])],
        ["legendary", "mythical", "best value", "worst value"], default="")
    d[["num", "name", "capture_rate", "rarity", "total", "highlight"]].sort_values("num") \
        .to_csv(OUT / "q1_rarity.csv", index=False)
    return res


# ---------------------------------------------------------------- Q2
def q2_size(df):
    res = {"weight_overall": correlation(df["weight_kg"], df["total"]),
           "height_overall": correlation(df["height_m"], df["total"]),
           "within_stage": {}}
    for stage, g in df.groupby("evo_stage"):
        res["within_stage"][int(stage)] = {
            "n": len(g),
            "weight": correlation(g["weight_kg"], g["total"]),
            "median_weight_kg": float(g["weight_kg"].median()),
            "median_total": float(g["total"].median())}
    res["stage_vs_weight"] = correlation(df["evo_stage"], df["weight_kg"])
    res["stage_vs_total"] = correlation(df["evo_stage"], df["total"])
    df[["num", "name", "height_m", "weight_kg", "total", "evo_stage"]] \
        .to_csv(OUT / "q2_size.csv", index=False)
    return res


# ---------------------------------------------------------------- Q3
def q3_speed(df):
    d = df.copy()
    d["toughness"] = d["hp"] + d["defense"] + d["sp_def"]
    fe, nfe = d[d["fully_evolved"]], d[~d["fully_evolved"]]
    # The trade-off holds within both groups; pooling them hides it (Simpson's paradox),
    # because fully evolved Pokémon are faster AND tougher on average.
    res = {"overall": correlation(d["speed"], d["toughness"]),
           "fully_evolved": correlation(fe["speed"], fe["toughness"]),
           "not_fully_evolved": correlation(nfe["speed"], nfe["toughness"]),
           "group_medians": {g: {"speed": float(x["speed"].median()), "toughness": float(x["toughness"].median())}
                             for g, x in (("fully_evolved", fe), ("not_fully_evolved", nfe))}}
    sq, tq = d["speed"].rank(pct=True), d["toughness"].rank(pct=True)
    both = (sq + tq) / 2
    d["highlight"] = np.select([(sq >= .75) & (tq >= .75), (sq <= .25) & (tq <= .25),
                                (sq >= .75) & (tq <= .25), (sq <= .25) & (tq >= .75)],
                               ["fast and tough", "slow and fragile",
                                "fast but fragile", "slow but tough"], default="")
    for label in ["fast and tough", "slow and fragile", "fast but fragile", "slow but tough"]:
        g = d[d["highlight"] == label]
        order = both if label in ("fast and tough",) else -both if label == "slow and fragile" \
            else (sq - tq) if label == "fast but fragile" else (tq - sq)
        res[label.replace(" ", "_")] = g.loc[order[g.index].sort_values(ascending=False).index,
                                             "name"].tolist()
    d[["num", "name", "speed", "toughness", "total", "fully_evolved", "highlight"]] \
        .to_csv(OUT / "q3_speed.csv", index=False)
    return res


# ---------------------------------------------------------------- Q4
def cramers_v(colours, types):
    table = pd.crosstab(colours, types).to_numpy().astype(float)
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))


def q4_colour(df):
    colours, type1 = df["color"].to_numpy(), df["type1"].to_numpy()
    res = {"association": shuffle_test(lambda c: cramers_v(c, type1), colours),
           "note": "Cramér's V between colour and primary type; 0 = no link, 1 = colour decides type"}

    # Headline pairs: either type slot counts
    has = {t: ((df["type1"] == t) | (df["type2"] == t)).to_numpy()
           for t in sorted(set(df["type1"]) | set(df["type2"]) - {""})}
    rows = []
    for colour in sorted(df["color"].unique()):
        in_colour = colours == colour
        for t, mask in has.items():
            count = int((in_colour & mask).sum())
            share, overall = count / in_colour.sum(), mask.mean()
            rows.append({"color": colour, "type": t, "count": count,
                         "color_n": int(in_colour.sum()), "share": round(share, 3),
                         "overall_share": round(overall, 3), "lift": round(share / overall, 2)})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "q4_colour_grid.csv", index=False)

    # Strongest colour→type pairs with enough Pokémon to mean something
    top = grid[(grid["count"] >= 4)].sort_values("lift", ascending=False).head(5)
    res["top_pairs"] = []
    for r in top.itertuples():
        mask = has[r.type]
        test = shuffle_test(lambda c, col=r.color, m=mask: float((m[c == col]).mean()), colours)
        rule_breakers = df.loc[(df["color"] == r.color) & ~mask, "name"].tolist()
        res["top_pairs"].append({"color": r.color, "type": r.type, "count": r.count,
                                 "color_n": r.color_n, "share": r.share,
                                 "overall_share": r.overall_share, "lift": r.lift,
                                 "p": test["p"], "rule_breakers": rule_breakers})
    df[["num", "name", "color", "type1", "type2"]].to_csv(OUT / "q4_colour.csv", index=False)
    return res


# ---------------------------------------------------------------- Q5
def q5_evolution(df):
    by_name = df.set_index("name")
    post = df[df["evolves_from"] != ""]
    pairs = []
    for r in post.itertuples():
        pre = by_name.loc[r.evolves_from]
        row = {"from": r.evolves_from, "to": r.name, "from_num": int(pre["num"]),
               "to_num": int(r.num), "stage_step": f"{int(pre['evo_stage'])}->{int(r.evo_stage)}",
               "from_total": int(pre["total"]), "to_total": int(r.total)}
        for s in STATS:
            row[f"{s}_gain_pct"] = round((getattr(r, s) / pre[s] - 1) * 100, 1)
        pairs.append(row)
    p = pd.DataFrame(pairs)
    assert len(p) == 72, len(p)
    p["gain"] = p["to_total"] - p["from_total"]
    p["gain_pct"] = (p["to_total"] / p["from_total"] - 1) * 100
    p = p.sort_values("gain_pct", ascending=False)
    p.round(1).to_csv(OUT / "q5_evolution.csv", index=False)

    stat_gain = {s: float(p[f"{s}_gain_pct"].median()) for s in STATS}
    return {"test": paired_shuffle_test(p["gain"]),
            "n_pairs": len(p),
            "median_gain_points": float(p["gain"].median()),
            "median_gain_pct": round(float(p["gain_pct"].median()), 1),
            "pairs_that_got_weaker": p.loc[p["gain"] <= 0, ["from", "to", "gain"]].to_dict("records"),
            "biggest_glow_ups": p.head(5)[["from", "to", "from_total", "to_total", "gain_pct"]]
                .round(1).to_dict("records"),
            "smallest_gains": p.tail(5)[["from", "to", "from_total", "to_total", "gain_pct"]]
                .round(1).to_dict("records"),
            "median_gain_pct_by_stat": {k: round(v, 1) for k, v in
                                        sorted(stat_gain.items(), key=lambda kv: -kv[1])}}


def main():
    OUT.mkdir(exist_ok=True)
    df = load()
    results = {"n_pokemon": len(df), "strength": "total of the six battle stats",
               "q1_rarity": q1_rarity(df), "q2_size": q2_size(df), "q3_speed": q3_speed(df),
               "q4_colour": q4_colour(df), "q5_evolution": q5_evolution(df)}

    # Appendix: correct the main p-values for running several tests at once
    main_tests = {
        "Q1 rarity vs strength": results["q1_rarity"]["overall"]["p"],
        "Q2 weight vs strength": results["q2_size"]["weight_overall"]["p"],
        "Q2 height vs strength": results["q2_size"]["height_overall"]["p"],
        **{f"Q2 weight vs strength, stage {s}": v["weight"]["p"]
           for s, v in results["q2_size"]["within_stage"].items()},
        "Q3 speed vs toughness": results["q3_speed"]["overall"]["p"],
        "Q3 speed vs toughness, fully evolved": results["q3_speed"]["fully_evolved"]["p"],
        "Q3 speed vs toughness, not fully evolved": results["q3_speed"]["not_fully_evolved"]["p"],
        "Q4 colour vs type": results["q4_colour"]["association"]["p"],
        "Q5 evolution gain": results["q5_evolution"]["test"]["p"],
    }
    adj = benjamini_hochberg(list(main_tests.values()))
    results["appendix_multiple_testing"] = {
        k: {"p": round(p, 5), "p_adjusted_BH": round(a, 5)}
        for (k, p), a in zip(main_tests.items(), adj)}

    path = OUT / "results.json"
    if path.exists():  # keep the model section written by train_model.py
        old = json.loads(path.read_text())
        if "model" in old:
            results["model"] = old["model"]
    path.write_text(json.dumps(results, indent=2, default=str, ensure_ascii=False))
    print(f"Wrote results to {OUT}")


if __name__ == "__main__":
    main()
