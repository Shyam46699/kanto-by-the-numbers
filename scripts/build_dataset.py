"""Merge the Kaggle stats file with PokéAPI data into one clean Gen 1 table.

Inputs:   data/raw/PokemonData.csv   (Kaggle: types, battle stats, legendary)
          data/raw/pokeapi/          (from scripts/fetch_pokeapi.py)
Output:   data/processed/gen1.csv    (151 rows, one per original Pokémon)

Choices made here:
- Mega Evolutions are dropped (they arrived in 2013, not 1996).
- `type1`/`type2` are the ORIGINAL 1996 types, taken from PokéAPI's
  `past_types`; today's types are kept as `modern_type1`/`modern_type2`.
- Evolution stage counts only Gen 1 Pokémon, so Pikachu is stage 1 even
  though Pichu (Gen 2) was added before it later.

Usage:  python3 scripts/build_dataset.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
API = RAW / "pokeapi"
OUT = ROOT / "data" / "processed" / "gen1.csv"

STAT_MAP = {"hp": "hp", "attack": "attack", "defense": "defense",
            "special-attack": "sp_atk", "special-defense": "sp_def", "speed": "speed"}
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9}


def load(endpoint, num):
    return json.loads((API / endpoint / f"{num}.json").read_text())


def type_names(types):
    names = [t["type"]["name"].capitalize() for t in sorted(types, key=lambda t: t["slot"])]
    return (names + [""])[:2]


def original_types(pokemon):
    """past_types lists the types a Pokémon had UP TO a generation;
    the entry with the earliest generation is its 1996 typing."""
    past = pokemon["past_types"]
    if not past:
        return type_names(pokemon["types"])
    earliest = min(past, key=lambda p: ROMAN[p["generation"]["name"].split("-")[1]])
    return type_names(earliest["types"])


def main():
    kaggle = {}
    with open(RAW / "PokemonData.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["Generation"] == "1" and "Mega" not in r["Name"]:
                if not r["Name"].isascii():  # Kaggle file double-encoded ♀/♂
                    r["Name"] = r["Name"].encode("cp1252").decode("utf-8")
                kaggle[int(r["Num"])] = r
    assert len(kaggle) == 151, len(kaggle)

    species = {n: load("pokemon-species", n) for n in range(1, 152)}
    pokemon = {n: load("pokemon", n) for n in range(1, 152)}
    name_to_num = {species[n]["name"]: n for n in species}

    # Evolution links, restricted to Gen 1
    parent = {}
    for n, s in species.items():
        pre = s["evolves_from_species"]
        if pre and pre["name"] in name_to_num:
            parent[n] = name_to_num[pre["name"]]
    has_child = set(parent.values())

    def stage(n):
        return 1 if n not in parent else 1 + stage(parent[n])

    rows, mismatches = [], []
    for n in range(1, 152):
        k, s, p = kaggle[n], species[n], pokemon[n]
        api_stats = {STAT_MAP[x["stat"]["name"]]: x["base_stat"] for x in p["stats"]}
        kag_stats = {"hp": int(k["HP"]), "attack": int(k["Attack"]), "defense": int(k["Defense"]),
                     "sp_atk": int(k["SpAtk"]), "sp_def": int(k["SpDef"]), "speed": int(k["Speed"])}
        if api_stats != kag_stats:
            mismatches.append((k["Name"], kag_stats, api_stats))

        t1, t2 = original_types(p)
        m1, m2 = type_names(p["types"])
        gr = s["gender_rate"]  # -1 = genderless, else eighths female
        rows.append({
            "num": n,
            "name": k["Name"],
            "type1": t1, "type2": t2,
            "modern_type1": m1, "modern_type2": m2,
            **kag_stats,
            "total": sum(kag_stats.values()),
            "legendary": k["Legendary"] == "TRUE",
            "mythical": s["is_mythical"],
            "height_m": p["height"] / 10,      # API gives decimetres
            "weight_kg": p["weight"] / 10,     # API gives hectograms
            "color": s["color"]["name"],
            "shape": s["shape"]["name"] if s["shape"] else "",
            "capture_rate": s["capture_rate"],  # 3 = hardest, 255 = easiest
            "genderless": gr == -1,
            "pct_female": "" if gr == -1 else round(gr / 8 * 100, 1),
            "egg_cycles": s["hatch_counter"],
            "base_happiness": s["base_happiness"],
            "growth_rate": s["growth_rate"]["name"],
            "evolves_from": kaggle[parent[n]]["Name"] if n in parent else "",
            "evo_stage": stage(n),
            "fully_evolved": n not in has_child,
            "evo_chain": int(s["evolution_chain"]["url"].rstrip("/").split("/")[-1]),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows x {len(rows[0])} columns to {OUT}")
    print(f"Stat mismatches Kaggle vs PokéAPI: {len(mismatches)}")
    for name, kag, api in mismatches:
        diff = {k: (kag[k], api[k]) for k in kag if kag[k] != api[k]}
        print(f"  {name}: {diff}  (kaggle, pokeapi)")


if __name__ == "__main__":
    main()
