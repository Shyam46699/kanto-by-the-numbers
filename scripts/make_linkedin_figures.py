"""Static chart images and cover for the LinkedIn version of the article.

Reads:   docs/data.json, docs/sprites.webp, assets/fonts/*.ttf (fonts: see README)
Writes:  linkedin/images/*.png  (1600 px wide figures, 1920 x 1080 cover)

Figures carry no title, only a legend where the marks need one; the article's captions name them.
Explanations live in the article text and captions; sources live in the references.
White background, EB Garamond titles, quiet sans labels.

Usage:  ~/.venvs/pokemon-ml/bin/python scripts/make_linkedin_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "linkedin" / "images"
D = json.loads((ROOT / "docs" / "data.json").read_text())
P, R = D["pokemon"], D["results"]
BY_NAME = {p["name"]: p for p in P}
SHEET = Image.open(ROOT / "docs" / "sprites.webp").convert("RGBA")
T = D["sprites"]["tile"]; COLS = D["sprites"]["cols"]

for f in (ROOT / "assets" / "fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(f))
SERIF, SANS = "EB Garamond", "Helvetica Neue"

C = dict(bg="#ffffff", ink="#161514", ink2="#4a4844", muted="#858379", grid="#e6e5e0",
         axis="#c3c2b7", accent="#9b1c1c", s1="#2a78d6", s2="#eb6834", s3="#1baf7a", dot="#c9c8c1")
# muted green ramp for the colour-by-type grid (white -> deep green)
SEQ = ["#ffffff", "#e9f1e5", "#cbdfc1", "#9fc292", "#5f9354", "#2f5e2a"]
plt.rcParams.update({
    "font.family": [SANS, "Arial Unicode MS", "DejaVu Sans"], "font.size": 10.5, "text.color": C["ink"],
    "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "axes.facecolor": C["bg"],
    "xtick.color": C["muted"], "ytick.color": C["muted"], "figure.facecolor": C["bg"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": C["grid"], "grid.linewidth": .8, "axes.axisbelow": True,
})


def sprite(n):
    i = n - 1
    return np.asarray(SHEET.crop(((i % COLS) * T, (i // COLS) * T, (i % COLS + 1) * T, (i // COLS + 1) * T)))


def frame(h=5.0, w=8.0):
    """Blank figure; figures carry no title (the article's caption names them).
    Returns the figure and the fraction of height where the plot area may start."""
    fig = plt.figure(figsize=(w, h), dpi=200)
    return fig, 1 - .1 / h


def callout(ax, names, label, dx, dy, zoom=.3):
    pts = [BY_NAME[n] for n in names]
    xy = (np.mean([ax._getx(p) for p in pts]), np.mean([ax._gety(p) for p in pts]))
    for k, n in enumerate(names):
        off = (k - (len(names) - 1) / 2) * 96 * zoom * .9
        ab = AnnotationBbox(OffsetImage(sprite(BY_NAME[n]["n"]), zoom=zoom), xy,
                            xybox=(dx + off, dy), boxcoords="offset points", frameon=False,
                            arrowprops=dict(arrowstyle="-", color=C["muted"], lw=.7) if k == len(names) // 2 else None)
        ax.add_artist(ab)
    ax.annotate(label, xy, xytext=(dx, dy - 96 * zoom / 2 - 4), textcoords="offset points", ha="center", va="top",
                size=8.5, color=C["ink"], zorder=5)


def tag(ax, name, xytext, ha="left"):
    """Small name label (placed in data coordinates) with a thin leader line, for dots without a sprite."""
    p = BY_NAME[name]
    ax.annotate(name, (ax._getx(p), ax._gety(p)), xytext=xytext, textcoords="data", ha=ha, va="center",
                size=8, color=C["ink2"], zorder=5,
                arrowprops=dict(arrowstyle="-", color=C["muted"], lw=.6, shrinkA=1, shrinkB=3))


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, facecolor=C["bg"])
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- permutation test (Q1)
def fig_shuffle():
    q, rho = D["q1_null"], R["q1_rarity"]["overall"]["rho"]
    fig, top = frame(h=3.5)
    ax = fig.add_axes([.07, .16, .88, top - .2])
    e = np.array(q["edges"]); ax.bar(e[:-1], q["counts"], width=np.diff(e) * .9, align="edge", color=C["dot"])
    ax.axvline(rho, color=C["accent"], lw=2)
    ax.text(rho, max(q["counts"]) * .95, "  Real data: %.2f" % rho, color=C["accent"], size=10, weight="bold", va="top")
    ax.text(0, max(q["counts"]) * 1.04, "10,000 shuffles", ha="center", size=9, color=C["ink2"])
    ax.set_xlim(-1, 1); ax.set_yticks([]); ax.grid(False)
    ax.spines["bottom"].set_visible(True)
    ax.set_xticks([-1, -.5, 0, .5, 1], ["−1", "−0.5", "0", "+0.5", "+1"])
    ax.set_xlabel("Correlation", labelpad=6)
    save(fig, "02-shuffle-test.png")


# ---------------------------------------------------------------- Q1
def fig_q1():
    fig, top = frame(h=4.5)
    ax = fig.add_axes([.08, .13, .88, top - .2])
    col = {"legendary": C["s1"], "mythical": C["s1"], "best value": C["s3"], "worst value": C["s2"]}
    jit = lambda p: ((p["n"] * 37) % 11 - 5) * 0.9
    ax._getx = lambda p: p["catch"] + jit(p); ax._gety = lambda p: p["total"]
    for layer in (False, True):
        pts = [p for p in P if (p["q1"] in col) == layer]
        ax.scatter([ax._getx(p) for p in pts], [p["total"] for p in pts], s=26,
                   c=[col.get(p["q1"], C["dot"]) for p in pts], edgecolors=C["bg"], linewidths=1.2, zorder=3 if layer else 2)
    ax.set_xlim(262, -6); ax.set_ylim(150, 720)
    ax.set_xticks([255, 190, 120, 45, 3]); ax.set_yticks(range(200, 701, 100))
    ax.set_xlabel("Catch rate   (← easy to catch · hard to catch →)", labelpad=6); ax.set_ylabel("Strength")
    callout(ax, ["Mewtwo"], "Mewtwo", -45, -8)
    callout(ax, ["Arcanine"], "Arcanine", 0, 34)
    callout(ax, ["Bulbasaur", "Charmander", "Squirtle"], "The starters", -60, -40)
    callout(ax, ["Magikarp"], "Magikarp", 45, 20)
    for n, xy, ha in [("Ninetales", (118, 540), "right"), ("Golduck", (118, 510), "right"), ("Muk", (118, 480), "right"),
                      ("Cloyster", (52, 578), "center"), ("Ditto", (22, 255), "left"), ("Dratini", (22, 345), "left")]:
        tag(ax, n, xy, ha)
    for lab, c in [("Legendary or mythical", C["s1"]), ("Best value for its catch rate", C["s3"]),
                   ("Worst value for its catch rate", C["s2"]), ("Everyone else", C["dot"])]:
        ax.scatter([], [], s=30, c=c, label=lab)
    ax.legend(loc="lower left", bbox_to_anchor=(-.01, 1.0), ncol=4, frameon=False, fontsize=8.5, handletextpad=.1, columnspacing=1)
    save(fig, "03-rarity.png")


# ---------------------------------------------------------------- Q2
def fig_q2():
    fig, top = frame(h=3.9)
    names = {1: "First stage", 2: "Second stage", 3: "Third stage"}; cols = {1: C["s1"], 2: C["s2"], 3: C["s3"]}
    for i, st in enumerate((1, 2, 3)):
        ax = fig.add_axes([.07 + i * .31, .16, .27, top - .26])
        pts = [p for p in P if p["stage"] == st]
        ax.scatter([p["w"] for p in pts], [p["total"] for p in pts], s=18, c=cols[st], edgecolors=C["bg"], linewidths=1)
        ax.set_xscale("log"); ax.set_xlim(.07, 700); ax.set_ylim(150, 720)
        ax.set_xticks([.1, 1, 10, 100], ["0.1", "1", "10", "100"]); ax.minorticks_off()
        ax.set_yticks(range(200, 701, 100)); ax.tick_params(labelsize=8.5)
        if i: ax.set_yticklabels([])
        else: ax.set_ylabel("Strength")
        if i == 1: ax.set_xlabel("Weight (kg, log scale)")
        ax.set_title(f"{names[st]} ({len(pts)} Pokémon)", size=9.5, loc="left", color=C["ink"])
        ax._getx = lambda p: p["w"]; ax._gety = lambda p: p["total"]
        if st == 1:
            callout(ax, ["Onix"], "Onix", 14, -48, zoom=.26)
    save(fig, "04-size.png")


# ---------------------------------------------------------------- Q3
def fig_q3():
    """Three panels: all 151 (coloured by group), then each group alone. The trade-off
    holds within both groups but disappears when they are pooled (Simpson's paradox)."""
    q = R["q3_speed"]
    fig, top = frame(h=4.1)
    grp = lambda p: C["s1"] if p["final"] else C["s2"]
    panels = [("All 151 Pokémon", P, None),
              (f"Not yet fully evolved ({q['not_fully_evolved']['n']})", [p for p in P if not p["final"]], C["s2"]),
              (f"Fully evolved ({q['fully_evolved']['n']})", [p for p in P if p["final"]], C["s1"])]
    for i, (ttl, pts, col) in enumerate(panels):
        ax = fig.add_axes([.07 + i * .31, .15, .27, top - .3])
        ax.scatter([p["speed"] for p in pts], [p["tough"] for p in pts], s=14,
                   c=[col or grp(p) for p in pts], edgecolors=C["bg"], linewidths=.8, zorder=3)
        ax.set_xlim(10, 158); ax.set_ylim(70, 375); ax.tick_params(labelsize=8)
        ax.set_xticks([25, 75, 125]); ax.set_title(ttl, size=9, loc="left")
        if i == 1: ax.set_xlabel("Speed")
        if i: ax.set_yticklabels([])
        else: ax.set_ylabel("Toughness (HP + Defence + Sp. Defence)", size=8.5)
        ax._getx = lambda p: p["speed"]; ax._gety = lambda p: p["tough"]
        if i == 2:
            callout(ax, ["Snorlax"], "Snorlax", 30, -6, zoom=.22)
            callout(ax, ["Alakazam"], "Alakazam", 2, -34, zoom=.22)
            callout(ax, ["Mewtwo"], "Mewtwo", -34, 4, zoom=.22)
    ax0 = fig.axes[0]
    ax0.scatter([], [], s=24, c=C["s2"], label="Not yet fully evolved"); ax0.scatter([], [], s=24, c=C["s1"], label="Fully evolved")
    ax0.legend(loc="lower left", bbox_to_anchor=(-.02, 1.09), ncol=2, frameon=False, fontsize=8.5, handletextpad=.1, columnspacing=1)
    save(fig, "05-speed-vs-toughness.png")


# ---------------------------------------------------------------- Q4
def fig_q4():
    types = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon"]
    g = {(r["color"], r["type"]): r for r in D["q4_grid"]}
    colours = sorted({r["color"] for r in D["q4_grid"]}, key=lambda c: -g[(c, "Normal")]["color_n"])
    strongest = {(t["color"], t["type"]) for t in R["q4_colour"]["top_pairs"]}
    sw = {"brown": "#9a6233", "blue": "#3d72d9", "purple": "#8b56b8", "yellow": "#efc935", "red": "#d6463c",
          "pink": "#f0a3c1", "green": "#4ea852", "gray": "#9b9a96", "white": "#f5f5f2", "black": "#262626"}
    fig, top = frame(h=4.8)
    ax = fig.add_axes([.2, .13, .77, top - .25]); ax.grid(False)
    step = lambda s: 0 if s == 0 else 1 if s <= .1 else 2 if s <= .2 else 3 if s <= .35 else 4 if s <= .5 else 5
    for yi, c in enumerate(colours):
        for xi, t in enumerate(types):
            r = g[(c, t)]; k = step(r["share"])
            ax.add_patch(Rectangle((xi + .06, yi + .08), .88, .84, color=SEQ[k], lw=0))
            if (c, t) in strongest:   # outline sits in the white gutter, so it reads on any shade
                ax.add_patch(Rectangle((xi + .01, yi + .03), .98, .94, fill=False, ec=C["ink"], lw=2))
            if r["count"]:
                ax.text(xi + .5, yi + .52, r["count"], ha="center", va="center", size=8, color="white" if k >= 4 else C["ink"])
        ax.add_patch(Rectangle((-2.75, yi + .28), .42, .44, color=sw[c], ec=C["axis"], lw=.6, clip_on=False))
        ax.text(-2.15, yi + .52, c.capitalize(), va="center", size=9, color=C["ink"])
        ax.text(-.2, yi + .52, str(g[(c, 'Normal')]['color_n']), va="center", ha="right", size=8, color=C["muted"])
    ax.set_xlim(0, len(types)); ax.set_ylim(len(colours), 0)
    ax.set_xticks([i + .5 for i in range(len(types))], types, rotation=45, ha="left", size=8.5)
    ax.xaxis.tick_top(); ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    # legend: shade key + outline key, below the grid
    lx, ly, s = .06, .045, .022
    fig.text(lx, ly + s * .8, "Share of the colour with this type:", size=8.5, color=C["ink2"], va="center")
    for k, lab in enumerate(["≤10%", "≤20%", "≤35%", "≤50%", ">50%"], start=1):
        x = lx + .24 + (k - 1) * .08
        fig.add_artist(Rectangle((x, ly), s * 1.3, s * 1.6, color=SEQ[k], ec=C["axis"] if k == 1 else "none", lw=.5, transform=fig.transFigure))
        fig.text(x + s * 1.3 + .006, ly + s * .8, lab, size=8, color=C["ink2"], va="center")
    ox = lx + .24 + 5 * .08 + .02
    fig.add_artist(Rectangle((ox, ly), s * 1.3, s * 1.6, fill=False, ec=C["ink"], lw=1.8, transform=fig.transFigure))
    fig.text(ox + s * 1.3 + .006, ly + s * .8, "strongest links", size=8, color=C["ink2"], va="center")
    save(fig, "06-colour-and-type.png")


# ---------------------------------------------------------------- Q5
def fig_q5():
    pairs = D["q5_pairs"]
    rows = pairs[:5] + [None] + pairs[-5:]
    fig, top = frame(h=4.7)
    ax = fig.add_axes([.42, .06, .47, top - .2]); ax.grid(True, axis="x"); ax.grid(False, axis="y")
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        if r is None:
            ax.text(450, y, "· · ·  62 more in between  · · ·", ha="center", va="center", size=8.5, color=C["muted"]); continue
        ax.plot([r["from_total"], r["to_total"]], [y, y], color=C["axis"], lw=2, zorder=1)
        ax.scatter([r["from_total"]], [y], s=40, c=C["dot"], edgecolors=C["bg"], linewidths=1.2, zorder=2)
        ax.scatter([r["to_total"]], [y], s=40, c=C["s1"], edgecolors=C["bg"], linewidths=1.2, zorder=2)
        ax.text(720, y, f"+{round(r['gain_pct'])}%", va="center", ha="right", size=9.5, weight="bold")
        ax_x = lambda fx: (fx - .42) / .47          # figure fraction -> axes fraction
        for k, num in enumerate((r["from_num"], r["to_num"])):
            ax.add_artist(AnnotationBbox(OffsetImage(sprite(num), zoom=.2), (ax_x(.065 + k * .038), y),
                                         xycoords=ax.get_yaxis_transform(), frameon=False, annotation_clip=False))
        ax.text(ax_x(.128), y, f"{r['from']} → {r['to']}", transform=ax.get_yaxis_transform(), va="center", size=9)
    ax.set_xlim(150, 725); ax.set_ylim(-.7, len(rows) - .3)
    ax.set_xticks([200, 300, 400, 500, 600]); ax.set_yticks([]); ax.tick_params(labelsize=8.5)
    ax.xaxis.tick_top(); ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Strength", labelpad=6); ax.xaxis.set_label_position("top")
    ax.scatter([], [], s=40, c=C["dot"], label="Before evolving"); ax.scatter([], [], s=40, c=C["s1"], label="After evolving")
    ax.legend(loc="lower left", bbox_to_anchor=(-.81, 1.0), ncol=2, frameon=False, fontsize=8.5, handletextpad=.1)
    save(fig, "07-evolution.png")


# ---------------------------------------------------------------- model
def fig_model():
    surprise = ["Onix", "Ditto", "Mew", "Mewtwo"]
    fig, top = frame(h=4.7)
    ax = fig.add_axes([.09, .12, .86, top - .16])
    ax.plot([150, 720], [150, 720], color=C["axis"], lw=1.2, zorder=1)
    ax.text(700, 690, "perfect prediction", ha="right", size=8.5, color=C["muted"])
    ax.scatter([p["total"] for p in P if p["name"] not in surprise], [p["ai"] for p in P if p["name"] not in surprise],
               s=22, c=C["s1"], edgecolors=C["bg"], linewidths=1, zorder=2)
    ax.scatter([BY_NAME[n]["total"] for n in surprise], [BY_NAME[n]["ai"] for n in surprise], s=30, c=C["s2"], edgecolors=C["bg"], linewidths=1, zorder=3)
    ax.set_xlim(150, 720); ax.set_ylim(150, 720); ax.set_xticks(range(200, 701, 100)); ax.set_yticks(range(200, 701, 100))
    ax.set_xlabel("Real strength"); ax.set_ylabel("Predicted strength"); ax.grid(True, axis="both")
    ax._getx = lambda p: p["total"]; ax._gety = lambda p: p["ai"]
    callout(ax, ["Onix"], "Onix", -70, 10, zoom=.28)
    callout(ax, ["Ditto"], "Ditto", -30, -40, zoom=.28)
    callout(ax, ["Mew"], "Mew", 0, -44, zoom=.28)
    callout(ax, ["Mewtwo"], "Mewtwo", -50, 22, zoom=.28)
    save(fig, "08-model-guesses.png")


def fig_importance():
    imp = R["model"]["importance"]
    rename = {"Rarity (catch rate)": "Catch rate", "Legendary": "Legendary or mythical"}
    fig, top = frame(h=3.1)
    ax = fig.add_axes([.27, .06, .63, top - .1]); ax.grid(False)
    labels = [rename.get(i["clue"], i["clue"]) for i in imp][::-1]; vals = [max(0, i["extra_error_points"]) for i in imp][::-1]
    ax.barh(range(len(vals)), vals, height=.55, color=C["s1"])
    for y, v in enumerate(vals):
        ax.text(v + .6, y, f"+{v:.1f} points" if v >= .5 else "no difference", va="center", size=9.5)
    ax.set_yticks(range(len(vals)), labels, size=10, color=C["ink"]); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False); ax.set_xlim(0, max(vals) * 1.25)
    save(fig, "09-what-the-model-relied-on.png")


# ---------------------------------------------------------------- data snapshot (model section)
def fig_snapshot():
    """Six rows of the exact table the model was trained on, printed the way Python
    (pandas) prints a wide table: middle colour columns elided with '...'."""
    import sys
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_model import features
    df = pd.read_csv(ROOT / "data" / "processed" / "gen1.csv")
    X, _ = features(df)
    X.index = df["name"]; X.index.name = None
    X["strength"] = df["total"].values
    X = X.loc[["Bulbasaur", "Pikachu", "Onix", "Magikarp", "Gyarados", "Mewtwo"]].round(2)
    left = ["log_height", "log_weight", "capture_rate", "evo_stage", "fully_evolved", "legendary",
            "colour_black", "colour_blue"]
    right = ["colour_yellow", "strength"]
    show = X[left].copy(); show["..."] = "..."; show = show.join(X[right])
    text = show.to_string(float_format=lambda v: f"{v:.2f}")
    text += f"\n\n[6 rows x {X.shape[1]} columns]   ({X.shape[1] - 1} inputs + strength, the target)"
    lines = text.split("\n")
    size = min(9.0, .95 * 8 * 72 / (.602 * max(map(len, lines))))   # monospace: fit the longest line to the width
    fig = plt.figure(figsize=(8, .35 + size * 1.55 / 72 * len(lines)), dpi=200)
    fig.text(.025, .5, text, family=["Menlo", "DejaVu Sans Mono"], size=size, va="center", ha="left",
             color=C["ink"], linespacing=1.55)
    save(fig, "10-data-snapshot.png")


# ---------------------------------------------------------------- cover
def cover():
    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), C["bg"])
    # sprite wall across the lower part, softly faded at the top edge
    cols, size = 24, 80
    wall = Image.new("RGBA", (cols * size, 4 * size), (0, 0, 0, 0))
    for k in range(cols * 4):
        n = (k * 7) % 151 + 1
        t = Image.fromarray(sprite(n)).resize((size - 6, size - 6), Image.LANCZOS)
        wall.alpha_composite(t, ((k % cols) * size + 3, (k // cols) * size + 3))
    fade = Image.linear_gradient("L").resize(wall.size)
    alpha = Image.eval(wall.getchannel("A"), lambda a: a)
    wall.putalpha(Image.composite(alpha, Image.new("L", wall.size, 0), fade))
    img.paste(wall, ((W - wall.width) // 2, H - wall.height - 20), wall)
    d = ImageDraw.Draw(img)
    gar = str(ROOT / "assets/fonts/EBGaramond[wght].ttf"); ita = str(ROOT / "assets/fonts/EBGaramond-Italic[wght].ttf")
    if not Path(gar).exists():
        print("skipped 01-cover.png: EB Garamond not found in assets/fonts/ (see README)")
        return
    f_title, f_sub, f_kick = ImageFont.truetype(gar, 132), ImageFont.truetype(ita, 54), ImageFont.truetype(gar, 30)
    d.line([(260, 108), (W - 260, 108)], fill=C["ink"], width=2); d.line([(260, 116), (W - 260, 116)], fill=C["ink"], width=2)
    # title centred as one line, with "Numbers" in the accent red (matches the interactive page)
    first, last = "Kanto by the ", "Numbers"
    x0 = W / 2 - f_title.getlength(first + last) / 2
    d.text((x0, 280), first, font=f_title, fill=C["ink"], anchor="lm")
    d.text((x0 + f_title.getlength(first), 280), last, font=f_title, fill=C["accent"], anchor="lm")
    d.text((W / 2, 420), "A statistical look at Gen 1 Pokémon", font=f_sub, fill=C["ink2"], anchor="mm")
    d.line([(260, 495), (W - 260, 495)], fill=C["ink"], width=1)
    img.save(OUT / "01-cover.png")
    print("wrote 01-cover.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cover(); fig_shuffle(); fig_q1(); fig_q2(); fig_q3(); fig_q4(); fig_q5(); fig_model(); fig_importance(); fig_snapshot()
