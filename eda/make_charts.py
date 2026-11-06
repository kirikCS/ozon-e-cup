"""Figures for the README and the technical report.

    python eda/make_charts.py [--data eda/chartdata.json] [--out docs/figures]

Every figure is rendered twice: `name.svg` for light backgrounds and `name-dark.svg`
for dark ones. The README switches between them with <picture>.

EDA figures read `chartdata.json`, a summary computed from `train.parquet` by
01_prep_and_checks.py to 05_chartdata.py. It is committed so that the figures rebuild
in a second without the 180 MB raw file. Target statistics are measured on the local
anchor 378 (2026-01-14) with a 30-day horizon, an exact copy of the competition setup:
the real target window 2026-02-14 to 2026-03-15 is not in the training data.

Result figures read the constants below; each block names where it was measured.
"""
import argparse, json, os
from datetime import date
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch

# Categorical slots are the validated reference palette: adjacent pairs keep ΔE ≥ 9.2
# (OKLab ×100) under deutan, protan and tritan simulation in both modes. Slot 3 sits
# below 3:1 on the light surface, so every mark in it carries a direct label.
THEMES = {
    "light": dict(suffix="", S1="#2a78d6", S2="#eb6834", S3="#1baf7a",
                  INK="#0b0b0b", SEC="#52514e", MUT="#898781", GRID="#e6e5e0",
                  SURF="#ffffff", NEUT="#cfcdc6",
                  SEQ=["#f3f7fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]),
    "dark": dict(suffix="-dark", S1="#3987e5", S2="#d95926", S3="#199e70",
                 INK="#f0efec", SEC="#c3c2b7", MUT="#898781", GRID="#262b33",
                 SURF="#0d1117", NEUT="#4a4945",
                 SEQ=["#161b22", "#184f95", "#2a78d6", "#86b6ef", "#cde2fb"]),
}

# ---------------------------------------------------------------- measured results
# Public leaderboard, best-so-far milestones in chronological order. The first ten are
# the submission log of the first two days, the rest are the commits that set a record.
LB = [("first blend", 1.6836226), ("LightGBM DART", 1.6723418),
      ("two-stage BTYD", 1.6577725), ("two-stage, log shift", 1.6570648),
      ("two-stage, log shift", 1.6546867), ("CatBoost", 1.6534269),
      ("blend of two", 1.6524999), ("OLS stack", 1.6508950),
      ("stack of 7", 1.6494144), ("stack of 9", 1.6491801),
      ("+ daily generative", 1.6470637), ("+ shape post-proc.", 1.6468118),
      ("stack_shape", 1.6466800), ("stack_final", 1.6464873),
      ("stack_v9", 1.6463025), ("stack_v11", 1.6462436), ("stack_v12", 1.6460012),
      ("v14", 1.6457758), ("v16", 1.6456395), ("v17", 1.6455060), ("v18", 1.6454945),
      ("v19", 1.6454440), ("v20", 1.6453935), ("v21", 1.6453455), ("v22", 1.6452810),
      ("v23_dir", 1.6452202)]
LB_PHASES = [(0, 9, "models and\nfirst stacks"), (10, 13, "daily\ngenerative"),
             (14, 16, "stack\nsolver"), (17, 25, "calibration ladder")]
BASELINE_LB = 2.19506

# Calibration ladder: public score of every step (stack_v9 to stack_v23_dir).
LADDER = [("v9", 1.6463024839), ("v11", 1.6462436395), ("v12", 1.6460012027),
          ("v14", 1.6457757641), ("v16", 1.6456395215), ("v17", 1.6455060133),
          ("v18", 1.6454944639), ("v19", 1.6454440380), ("v20", 1.6453934846),
          ("v21", 1.6453455262), ("v22", 1.6452809664), ("v23_dir", 1.6452202009)]
PENALTY = 3.29e-05        # expected public-minus-private gap per fitted parameter, SE²/R0

# Readout of the daily generative model on the local anchor: Monte-Carlo paths per
# customer against the exact Gauss-Hermite / Frullani quadrature.
READOUT_N = [96, 384, 1536, 6144]
READOUT_RMSLE = [1.73888, 1.73777, 1.73729, 1.73728]
READOUT_EXACT = 1.73723

# Best single vector on the local anchor: survival-integral threshold ladder.
BEST_SINGLE_LOCAL = 1.73556

# Gain curve on a held-out month with known outcome (2025-12-16 to 2026-01-14):
# share of next-month GMV held by the top x% of customers ranked by each score.
GAIN_X = [0, 5, 10, 20, 30, 50]
GAIN = {"perfect ranking": [0, 48.9, 66.0, 84.0, 92.9, 99.5],
        "our model": [0, 29.7, 43.7, 61.5, 73.4, 88.1],
        "last month's GMV": [0, 26.1, 39.5, 56.8, 68.8, 84.0]}


def setup(T):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": T["SURF"], "axes.facecolor": T["SURF"],
        "savefig.facecolor": T["SURF"], "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.edgecolor": T["GRID"],
        "axes.linewidth": 0.8, "axes.titlelocation": "left", "axes.titlepad": 9,
        "axes.labelcolor": T["SEC"], "axes.titlecolor": T["INK"], "text.color": T["INK"],
        "xtick.color": T["MUT"], "ytick.color": T["MUT"], "xtick.labelsize": 8,
        "ytick.labelsize": 8, "grid.color": T["GRID"], "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": T["SEC"],
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "svg.fonttype": "path", "svg.hashsalt": "ecup", "figure.dpi": 110,
    })


def frame(ax, T, axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(T["GRID"])
    ax.set_axisbelow(True)
    if axis:
        ax.grid(axis=axis)
    ax.tick_params(length=0)


def save(fig, T, out, name):
    fig.tight_layout()
    path = os.path.join(out, name + T["suffix"] + ".svg")
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print("   ", os.path.basename(path))


def plain(labels):
    return [l.replace("\u2013", "-") for l in labels]


def luminance(c):
    r, g, b = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in to_rgb(c)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ------------------------------------------------------------------------ EDA figures
def fig_target(D, T, out):
    td = D["target_dist"]
    lab = plain(D["transition"]["labels"])                # same eleven buckets
    shu = [r["share"] * 100 for r in td]
    shg = [r["gmv_share"] * 100 for r in td]
    fig, ax = plt.subplots(figsize=(7.8, 3.4))
    x = np.arange(len(lab)); w = 0.38
    ax.bar(x - w / 2 - 0.02, shu, w, color=T["S1"], label="share of customers")
    ax.bar(x + w / 2 + 0.02, shg, w, color=T["S2"], label="share of GMV")
    ax.set_xticks(x); ax.set_xticklabels(lab)
    ax.set_ylabel("%"); ax.set_xlabel("customer GMV over the 30-day horizon")
    ax.set_title("45.9% of customers buy nothing; 63 customers bring 2.5% of all GMV")
    ax.annotate(f"{shu[0]:.1f}% of customers,\n0% of GMV", (0.2, shu[0]), (0.75, shu[0] - 4),
                color=T["SEC"], fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=T["MUT"], lw=0.8))
    ax.legend(loc="upper right"); frame(ax, T)
    save(fig, T, out, "01_target")


def fig_lorenz(D, T, out):
    lz = D["lorenz"]
    pop = np.array(lz["pop"]) * 100; gmv = np.array(lz["gmv"]) * 100
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot([0, 100], [0, 100], lw=0.9, color=T["MUT"])
    ax.annotate("equal spending", (70, 57), color=T["MUT"], fontsize=8, rotation=31)
    ax.plot(100 - pop[::-1], 100 - gmv[::-1], color=T["S1"])
    con = {r["top_share_users"]: r["gmv_share"] for r in D["concentration"]}
    for f_, t_ in ((0.01, "top 1%"), (0.1, "top 10%"), (0.2, "top 20%")):
        ax.plot([f_ * 100], [con[f_] * 100], "o", ms=6, color=T["S1"],
                markeredgecolor=T["SURF"], markeredgewidth=1.5)
        ax.annotate(f"{t_}: {con[f_]*100:.0f}% of GMV", (f_ * 100, con[f_] * 100),
                    (7, -4), textcoords="offset points", color=T["SEC"], fontsize=8.5)
    ax.set_xlabel("top N% of customers by GMV"); ax.set_ylabel("share of GMV, %")
    ax.set_title("The top 10% of customers bring 53% of GMV")
    frame(ax, T)
    save(fig, T, out, "02_lorenz")


def fig_daily(D, T, out):
    # Two panels rather than two scales on one axis: daily GMV and its 30-day sum
    # differ thirtyfold.
    dd = D["daily"]; rr = D["roll30"]
    dt = np.array(dd["date"], dtype="datetime64[D]")
    rt = np.array(rr["date"], dtype="datetime64[D]")
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 4.6), sharex=True)
    axes[0].plot(dt, np.array(dd["gmv"]) / 1e3, lw=0.9, color=T["S1"])
    axes[0].set_ylabel("daily GMV, k")
    axes[0].set_title("Daily GMV doubled over 409 days, with a December peak and a January dip")
    axes[1].plot(rt, np.array(rr["gmv"]) / 1e6, lw=2.0, color=T["S1"])
    axes[1].set_ylabel("30-day GMV, M")
    axes[1].set_xlabel("bottom panel is the target quantity itself: a 30-day rolling sum")
    for ax in axes:
        ax.axvline(np.datetime64("2025-04-01"), color=T["MUT"], lw=1.0)
        ax.axvspan(np.datetime64("2026-01-15"), np.datetime64("2026-02-13"),
                   color=T["S3"], alpha=0.22, lw=0)
        ax.axvspan(np.datetime64("2026-02-14"), np.datetime64("2026-03-15"),
                   color=T["S2"], alpha=0.18, lw=0)
        ax.set_xlim(np.datetime64("2025-01-01"), np.datetime64("2026-03-20"))
        frame(ax, T)
    hi = axes[0].get_ylim()[1]
    axes[0].annotate("April 2025:\nchannel break", (np.datetime64("2025-04-06"), hi),
                     color=T["SEC"], fontsize=8.5, va="top")
    lo1 = axes[1].get_ylim()[0]
    axes[1].annotate("validation\ntarget", (np.datetime64("2026-01-12"), lo1), (0, 4),
                     textcoords="offset points", color=T["SEC"], fontsize=8.5,
                     ha="right", va="bottom")
    axes[1].annotate("forecast\nwindow", (np.datetime64("2026-02-16"), lo1), (0, 4),
                     textcoords="offset points", color=T["SEC"], fontsize=8.5,
                     ha="left", va="bottom")
    save(fig, T, out, "03_daily")


def fig_growth(D, T, out):
    mo = [r for r in D["monthly"] if r["days"] >= 28]       # 2026-02 is cut by the anchor
    ms = [date.fromisoformat(r["month"] + "-01").strftime("%b %y") for r in mo]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    for ax, vals, ttl in (
            (axes[0], [r["gmv"] / 1e6 for r in mo], "monthly GMV, M"),
            (axes[1], [r["uniq_buyers"] / 1e3 for r in mo], "buyers per month, k"),
            (axes[2], [r["aov"] for r in mo], "average order value")):
        ax.plot(ms, vals, color=T["S1"], marker="o", ms=3.5)
        ax.set_title(ttl, fontsize=9, fontweight="normal", color=T["SEC"])
        ax.set_xticks(ms[::3]); frame(ax, T)
    fig.suptitle("Buyers per month grew from 91k to 146k while the average order value fell",
                 x=0.005, ha="left", fontsize=10.5, fontweight="bold", color=T["INK"])
    save(fig, T, out, "04_growth")


def fig_recency(D, T, out):
    br = D["by_recency"]
    lb = [("0 d" if r["recency_bucket"] == "0-0"
           else r["recency_bucket"] + " d") for r in br]
    pp = [r["p_target_pos"] * 100 for r in br]
    nn = [r["n_users"] for r in br]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.bar(lb, pp, color=T["S1"], width=0.6)
    for i, (p_, n_) in enumerate(zip(pp, nn)):
        ax.annotate(f"{p_:.1f}%", (i, p_), (0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=T["SEC"])
        ax.annotate(f"{n_:,}", (i, 1.5), ha="center", fontsize=8, color="#ffffff")
    ax.set_ylabel("bought within 30 days, %")
    ax.set_xlabel("days since last visit at the anchor (inside the bar: customers)")
    ax.set_title("70% of anchor-day visitors buy within 30 days")
    frame(ax, T)
    save(fig, T, out, "05_recency")


def fig_transition(D, T, out):
    tr = D["transition"]
    M = np.array(tr["matrix"]) * 100
    cmap = LinearSegmentedColormap.from_list("seq", T["SEQ"])
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.grid(False)
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=M.max())
    lab = plain(tr["labels"])
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right")
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
    ax.set_xlabel("GMV over the next 30 days"); ax.set_ylabel("GMV over the previous 30 days")
    ax.set_title("72% of zero-GMV customers stay at zero next month")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] >= 4:
                cell = cmap(M[i, j] / M.max())
                ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="#0b0b0b" if luminance(cell) > 0.3 else "#ffffff")
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cb.set_label("% of row", color=T["SEC"], fontsize=8.5)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0, labelsize=8, colors=T["MUT"])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, T, out, "06_transition")


def fig_blocks(D, T, out):
    bl = D["blocks"]
    nm = [r["block"] for r in bl]
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.axvspan(9.5, len(nm) - 0.5, color=T["S3"], alpha=0.18, lw=0)
    ax.plot(nm, [r["share_active"] * 100 for r in bl], color=T["S1"], marker="o", ms=4)
    ax.plot(nm, [r["share_buyers"] * 100 for r in bl], color=T["S2"], marker="o", ms=4)
    ax.annotate("visited", (0, bl[0]["share_active"] * 100), (4, 7),
                textcoords="offset points", color=T["SEC"], fontsize=8.5)
    ax.annotate("bought", (0, bl[0]["share_buyers"] * 100), (4, 7),
                textcoords="offset points", color=T["SEC"], fontsize=8.5)
    ax.set_ylim(0, 112); ax.set_ylabel("share of customers, %")
    ax.set_xlabel("30-day block, B-0 = the last block before 2026-02-13")
    ax.set_title("All customers are active in each of the last three 30-day blocks")
    ax.annotate("selection window: 100.0000%,\nwhile B-3 is already 93.3%",
                (len(nm) - 1.2, 100), (-14, -34), textcoords="offset points",
                ha="right", color=T["SEC"], fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=T["MUT"], lw=0.8))
    frame(ax, T)
    save(fig, T, out, "07_blocks")


def fig_baselines(D, T, out):
    EN = {"const 0": "constant 0",
          "optimal constant (8.4)": "optimal constant (8.4)",
          "prev30 (naive AR = sample_submit)": "prev30 as is (organizers' sample)",
          "prev60 / 2": "prev60 / 2", "prev90 / 3": "prev90 / 3", "prev180 / 6": "prev180 / 6",
          "mean(prev30, prev60/2, prev90/3)": "mean of the three windows",
          "alpha*prev30 (alpha=0.30)": "0.30 × prev30",
          "lifetime mean per 30d": "lifetime mean per 30 days",
          "expm1(0.54*log1p(prev30)+0.95)": "log-linear regression on prev30",
          "oracle: mean log-target per prev30 bucket": "oracle: prev30 bucket",
          "oracle: prev30 bucket x recency bucket": "oracle: prev30 × recency"}
    bs = sorted([(EN.get(r["baseline"], r["baseline"]), r["rmsle"]) for r in D["baselines"]],
                key=lambda t: -t[1])
    bs.append(("best single model (ours)", BEST_SINGLE_LOCAL))
    names = [b[0] for b in bs]; vals = [b[1] for b in bs]
    cols = [T["S2"] if n.startswith("oracle") else T["NEUT"] for n in names]
    cols[-1] = T["S1"]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.barh(names, vals, color=cols, height=0.64)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.4f}", (v, i), (5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=T["SEC"])
    ax.set_xlim(1.55, 3.42)
    ax.set_xlabel("RMSLE on the local validation anchor (2026-01-14), lower is better")
    ax.legend(handles=[Patch(color=T["NEUT"], label="simple rule"),
                       Patch(color=T["S2"], label="in-sample oracle on buckets"),
                       Patch(color=T["S1"], label="our model")],
              loc="upper right")
    frame(ax, T, axis="x")
    fig.suptitle("In-sample oracles on aggregate buckets stop at 1.87",
                 x=0.005, ha="left", fontsize=10.5, fontweight="bold", color=T["INK"])
    save(fig, T, out, "08_baselines")


def fig_ladder(D, T, out):
    nm = [l[0] for l in LADDER]; sc = np.array([l[1] for l in LADDER])
    step = -np.diff(sc); netstep = step - PENALTY; net = np.cumsum(netstep)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    axes[0].plot(nm, sc, color=T["S1"], marker="o", ms=4)
    axes[0].set_ylabel("public RMSLE")
    axes[0].set_title("Public RMSLE per ladder step")
    axes[1].bar(nm[1:], netstep * 1e4, width=0.6,
                color=[T["S1"] if v > 0 else T["S2"] for v in netstep])
    axes[1].plot(nm[1:], net * 1e4, color=T["S3"], marker="o", ms=3.5, lw=1.6)
    axes[1].axhline(0, color=T["MUT"], lw=0.9)
    axes[1].set_ylim(top=max(net.max(), netstep.max()) * 1e4 * 1.30)
    axes[1].annotate(f"cumulative +{net[-1]*1e4:.2f}·10⁻⁴", (len(nm) - 2, net[-1] * 1e4), (-6, 8),
                     textcoords="offset points", ha="right", color=T["SEC"], fontsize=8.5)
    axes[1].annotate("v18 did not pay\nfor its penalty", (5, netstep[5] * 1e4), (8, 40),
                     textcoords="offset points", color=T["SEC"], fontsize=8.5,
                     arrowprops=dict(arrowstyle="-", color=T["MUT"], lw=0.8))
    axes[1].set_ylabel("×10⁻⁴ RMSLE")
    axes[1].set_title("Net gain per step after the 3.29·10⁻⁵ fitting penalty")
    axes[1].legend(handles=[Patch(color=T["S1"], label="net step"),
                            plt.Line2D([], [], color=T["S3"], marker="o", ms=3.5,
                                       label="cumulative")], loc="upper left", ncol=2)
    for ax in axes:
        ax.tick_params(axis="x", rotation=45); frame(ax, T)
    save(fig, T, out, "09_ladder")


def fig_dow(D, T, out):
    dw = D["dow"]
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    idx = np.array([r["gmv_index"] for r in dw]) * 100 - 100
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.bar(names, idx, color=[T["S1"] if v >= 0 else T["S2"] for v in idx], width=0.6)
    for i, v in enumerate(idx):
        ax.annotate(f"{v:+.1f}", (i, v), (0, 4 if v >= 0 else -13),
                    textcoords="offset points", ha="center", fontsize=8, color=T["SEC"])
    ax.axhline(0, color=T["MUT"], lw=0.9)
    ax.set_ylim(idx.min() - 1.4, idx.max() + 1.0)
    ax.set_ylabel("GMV vs. weekly mean, %")
    ax.set_title("GMV varies by about 2% across weekdays")
    frame(ax, T)
    save(fig, T, out, "10_dow")


# --------------------------------------------------------------------- result figures
def fig_progress(D, T, out):
    sc = np.array([s for _, s in LB]); x = np.arange(len(LB))
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    for k, (a, b, lab) in enumerate(LB_PHASES):
        if k % 2:
            ax.axvspan(a - 0.5, b + 0.5, color=T["NEUT"], alpha=0.25, lw=0)
        ax.annotate(lab, ((a + b) / 2, 1.6878), ha="center", va="top",
                    fontsize=8, color=T["SEC"])
    ax.step(x, sc, where="post", color=T["S1"], lw=2.0)
    ax.plot(x, sc, "o", ms=4, color=T["S1"], markeredgecolor=T["SURF"], markeredgewidth=1.2)
    for i, dx, dy in ((0, 4, 5), (5, 6, 4), (9, 6, 6), (16, -2, 10)):
        ax.annotate(f"{LB[i][0]}  {sc[i]:.4f}", (i, sc[i]), (dx, dy),
                    textcoords="offset points", fontsize=8.5, color=T["SEC"])
    ax.annotate(f"final  {sc[-1]:.7f}", (len(sc) - 1, sc[-1]), (0, -14),
                textcoords="offset points", ha="right", fontsize=9, fontweight="bold",
                color=T["INK"])
    ax.set_xlim(-0.6, len(sc) - 0.4); ax.set_ylim(1.6435, 1.6882)
    ax.set_xticks([])
    ax.set_xlabel("record-setting submissions in chronological order")
    ax.set_ylabel("public RMSLE")
    ax.set_title(f"Public leaderboard from {sc[0]:.4f} to {sc[-1]:.4f}"
                 f" (organizers' baseline {BASELINE_LB:.4f} is off the scale)")
    frame(ax, T)
    save(fig, T, out, "11_lb_progress")


def fig_readout(D, T, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.axhline(READOUT_EXACT, color=T["S2"], lw=1.6)
    ax.annotate(f"exact quadrature  {READOUT_EXACT:.5f}", (96, READOUT_EXACT), (0, 5),
                textcoords="offset points", fontsize=8.5, color=T["SEC"])
    ax.plot(READOUT_N, READOUT_RMSLE, color=T["S1"], marker="o", ms=6,
            markeredgecolor=T["SURF"], markeredgewidth=1.5)
    ax.annotate(f"96 paths (what we had):\n+{(READOUT_RMSLE[0]-READOUT_EXACT)*1e3:.2f}·10⁻³",
                (96, READOUT_RMSLE[0]), (14, -2), textcoords="offset points",
                fontsize=8.5, color=T["SEC"], va="center")
    ax.set_xscale("log", base=4)
    ax.set_xticks(READOUT_N); ax.set_xticklabels([f"{n:,}" for n in READOUT_N])
    ax.minorticks_off()
    ax.set_xlabel("Monte-Carlo paths per customer")
    ax.set_ylabel("RMSLE, local validation")
    ax.set_title("Monte-Carlo readout converges to the exact quadrature")
    frame(ax, T)
    save(fig, T, out, "12_readout")


def fig_gain(D, T, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([0, 50], [0, 50], lw=0.9, color=T["MUT"])
    ax.annotate("random", (40, 36), color=T["MUT"], fontsize=8)
    style = {"perfect ranking": (T["MUT"], 1.4), "our model": (T["S1"], 2.2),
             "last month's GMV": (T["S2"], 2.0)}
    nudge = {"perfect ranking": 0, "our model": 5, "last month's GMV": -5}
    for name, ys in GAIN.items():
        c, lw = style[name]
        ax.plot(GAIN_X, ys, color=c, lw=lw, marker="o", ms=4, label=name)
        ax.annotate(name, (GAIN_X[-1], ys[-1]), (6, nudge[name]), textcoords="offset points",
                    va="center", fontsize=8.5, color=T["SEC"])
    ax.axvline(10, color=T["GRID"], lw=1.0)
    ax.annotate("top 10%: 43.7% of GMV\nvs. 39.5% by last month", (10, 43.7), (8, -22),
                textcoords="offset points", fontsize=8.5, color=T["SEC"],
                arrowprops=dict(arrowstyle="-", color=T["MUT"], lw=0.8))
    ax.set_xlim(0, 64); ax.set_ylim(0, 104)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("top x% of customers, ranked by score")
    ax.set_ylabel("share of next-month GMV, %")
    ax.set_title("Gain curve on a held-out month")
    ax.legend(loc="lower right")
    frame(ax, T, axis="both")
    save(fig, T, out, "13_gain")


FIGURES = [fig_target, fig_lorenz, fig_daily, fig_growth, fig_recency, fig_transition,
           fig_blocks, fig_baselines, fig_ladder, fig_dow, fig_progress, fig_readout, fig_gain]


def main():
    H = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(H, "chartdata.json"))
    ap.add_argument("--out", default=os.path.join(H, "..", "docs", "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = json.load(open(a.data))
    for name, T in THEMES.items():
        print(f"{name} theme from {os.path.basename(a.data)}")
        setup(T)
        for f in FIGURES:
            f(D, T, a.out)


if __name__ == "__main__":
    main()
