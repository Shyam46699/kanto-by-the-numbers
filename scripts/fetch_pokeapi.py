"""Download PokéAPI data for the original 151 Pokémon.

Saves one JSON file per request under data/raw/pokeapi/ so we only ever
download once. Re-running skips files that already exist.

Usage:  python3 scripts/fetch_pokeapi.py
"""
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "pokeapi"
BASE = "https://pokeapi.co/api/v2"
HEADERS = {"User-Agent": "pokemon-ml-project/0.1"}


def fetch(endpoint: str, num: int) -> None:
    path = OUT / endpoint / f"{num}.json"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{BASE}/{endpoint}/{num}", headers=HEADERS)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            path.write_text(json.dumps(data))
            time.sleep(0.1)  # be polite to a free API
            return
        except Exception as exc:  # retry on network hiccups
            if attempt == 2:
                raise RuntimeError(f"{endpoint}/{num} failed: {exc}") from exc
            time.sleep(2)


def main() -> None:
    for num in range(1, 152):
        fetch("pokemon", num)          # height, weight, types, sprites
        fetch("pokemon-species", num)  # colour, catch rate, gender, hatch, evolution
        if num % 25 == 0:
            print(f"{num}/151 done")
    print("All 151 downloaded to", OUT)


if __name__ == "__main__":
    main()
