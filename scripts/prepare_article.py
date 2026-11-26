"""Prepare everything the article page loads.

1. sprites.webp  - one sprite sheet of the original Red/Blue artwork for all
                   151 Pokémon (Mew uses Red/Green; it has no Red/Blue image).
                   Tile i (0-based, Pokédex number - 1) sits at column i % 16,
                   row i // 16; each tile is 96 x 96 px.
2. data.json     - the numbers and per-Pokémon rows every chart needs.

Reads:  data/raw/gen1-images.zip (Kaggle image set, see README), data/processed/gen1.csv, results/*
Writes: docs/sprites.webp, docs/data.json  (docs/ is the interactive page)

Usage:  ~/.venvs/pokemon-ml/bin/python scripts/prepare_article.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_tools import N_SHUFFLES, SEED, _rowwise_pearson  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data" / "raw" / "gen1-images.zip"
OUT = ROOT / "docs"
TILE, COLS = 96, 16
FOLDER = {"Nidoran♀": "Nidoran-f", "Nidoran♂": "Nidoran-m",
          "Mr. Mime": "Mr-Mime", "Farfetch'd": "Farfetch"}


def sprite_sheet(df):
    z = zipfile.ZipFile(ZIP)
    paths = z.namelist()
    rows = -(-len(df) // COLS)
    sheet = Image.new("RGBA", (COLS * TILE, rows * TILE), (0, 0, 0, 0))
    for r in df.itertuples():
        folder = FOLDER.get(r.name, r.name)
        found = None
        for suffix in ("_RB.png", "_RG.png"):
            hits = [p for p in paths if f"/{folder}/" in p and p.endswith(suffix)]
            if hits:
                found = hits[0]
                break
        assert found, r.name
        im = Image.open(io.BytesIO(z.read(found))).convert("RGBA")
        im.thumbnail((TILE, TILE), Image.LANCZOS)
        i = r.num - 1
        x = (i % COLS) * TILE + (TILE - im.width) // 2
        y = (i // COLS) * TILE + (TILE - im.height) // 2
        sheet.alpha_composite(im, (x, y))
    sheet.save(OUT / "sprites.webp", "WEBP", quality=82, method=6)
    return {"tile": TILE, "cols": COLS, "rows": rows,
            "width": COLS * TILE, "height": rows * TILE}


def q1_null_histogram(df):
    """The 10,000 shuffled correlations for Q1, binned, for the explainer."""
    rng = np.random.default_rng(SEED)
    rx = rankdata(255 - df["capture_rate"])
    ry = rankdata(df["total"])
    shuffled = np.array([rng.permutation(ry) for _ in range(N_SHUFFLES)])
    null = _rowwise_pearson(np.broadcast_to(rx, shuffled.shape), shuffled)
    edges = np.linspace(-1, 1, 81)
    counts, _ = np.histogram(null, bins=edges)
    return {"edges": edges.round(3).tolist(), "counts": counts.tolist(),
            "max_abs_chance": round(float(np.abs(null).max()), 3)}


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/gen1.csv", keep_default_na=False)
    res = json.loads((ROOT / "results/results.json").read_text())
    q1 = pd.read_csv(ROOT / "results/q1_rarity.csv", keep_default_na=False)
    q3 = pd.read_csv(ROOT / "results/q3_speed.csv", keep_default_na=False)
    pred = pd.read_csv(ROOT / "results/model_predictions.csv")
    q4 = pd.read_csv(ROOT / "results/q4_colour_grid.csv")
    q5 = pd.read_csv(ROOT / "results/q5_evolution.csv")

    d = df.merge(q1[["num", "highlight"]].rename(columns={"highlight": "q1"}), on="num") \
          .merge(q3[["num", "toughness", "highlight"]].rename(columns={"highlight": "q3"}), on="num") \
          .merge(pred[["num", "ai_guess", "miss"]], on="num")
    pokemon = [{
        "n": int(r.num), "name": r.name, "t1": r.type1, "t2": r.type2, "color": r.color,
        "total": int(r.total), "catch": int(r.capture_rate), "h": float(r.height_m),
        "w": float(r.weight_kg), "stage": int(r.evo_stage), "final": str(r.fully_evolved) == "True",
        "speed": int(r.speed), "tough": int(r.toughness), "ai": int(r.ai_guess),
        "legendary": str(r.legendary) == "True", "mythical": str(r.mythical) == "True",
        "q1": r.q1, "q3": r.q3,
    } for r in d.itertuples()]

    data = {
        "sprites": sprite_sheet(df),
        "pokemon": pokemon,
        "results": res,
        "q1_null": q1_null_histogram(df),
        "q4_grid": q4.to_dict("records"),
        "q5_pairs": q5[["from", "to", "from_num", "to_num", "from_total", "to_total",
                        "gain_pct"]].to_dict("records"),
    }
    (OUT / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    for f in ("sprites.webp", "data.json"):
        print(f, round((OUT / f).stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
