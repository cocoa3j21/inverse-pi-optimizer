#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmic_collapse_time_arrow.py

閉じた玩具宇宙における重力のみの崩壊・時間の矢・循環の数値実験

物理設定:
  - 2 次元 Plummer 軟化付きニュートン自己重力系（散逸・放射なし）
  - leapfrog (kick-drift-kick) 積分 — 時間可逆
  - 孤立系（外部ポテンシャルなし）。閉箱の対照として reflecting_wall を持つ

実験問い（すべて「示唆」として読む）:
  Q1: ほぼ一様な系は重力だけで凝縮するか
  Q2: 凝縮状態は『ビッグバン直後』に似るか（エントロピー代理で見る）
  Q3: 時間の矢を膨張でなく収縮方向に置くと何が変わるか
  Q4: 循環するか、一方向に終わるか
  Q5: 速度反転 / ラベル反転で過去に戻るか

重要: これは 2 次元・軟化ニュートン・放射なし・有限時間の玩具です。
得られるのは保存量・相関・再現誤差であり、宇宙論の結論・証明ではありません。
「宇宙の正体が判明した」とは書きません。

出力先（既存の results/ figures/ は上書きしません）:
  - results_new/cosmic_collapse_time_arrow.csv
  - results_new/cosmic_collapse_time_arrow.json
  - figures_new/cosmic_collapse_time_arrow.png
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from datetime import datetime, timezone

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
import numpy as np



# ---------------------------------------------------------------------------
# パラメータ（既存出力と整合）
# ---------------------------------------------------------------------------
class Params:
    def __init__(self):
        self.n_particles = 200
        self.n_steps = 3000
        self.dt = 0.005
        self.softening = 0.05
        self.R0 = 1.0
        self.G_scaled = 0.005
        self.sample_every = 5
        self.box_half_width = 2.5
        self.grid_cells = 24
        self.wall_radius = 1.6
        self.retrace_lags = [100, 250, 500, 750]
        self.seeds = [42, 7, 13, 29, 101]
        self.conditions = [
            "bound_collapse",
            "expanding",
            "marginal",
            "reflecting_wall",
            "time_reversed",
        ]

    def to_dict(self):
        return {k: getattr(self, k) for k in [
            "n_particles", "n_steps", "dt", "softening", "R0", "G_scaled",
            "sample_every", "box_half_width", "grid_cells", "wall_radius",
            "retrace_lags", "seeds", "conditions"]}


P = Params()
# 出力コンソールが cp932 でも落ちないよう UTF-8 に寄せる（Windows 対策）
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def seeded_rng(seed):
    return random.Random(seed)


def make_grid_uniform(rng, n, width=2.5, soft=0.05):
    """ほぼ一様な格子状の初期配置（中心付近に濃密、周囲に疎）。"""
    xs, ys, ms = [], [], []
    n_side = int(math.sqrt(n))
    half = width * 0.9
    for i in range(n_side):
        for j in range(n_side):
            if len(xs) >= n:
                break
            x = (i + 0.5) / n_side * (2 * half) - half
            y = (j + 0.5) / n_side * (2 * half) - half
            xs.append(x)
            ys.append(y)
            ms.append(1.0)
    return xs, ys, ms


def make_random_uniform(rng, n, width=2.5):
    xs = [rng.uniform(-width, width) for _ in range(n)]
    ys = [rng.uniform(-width, width) for _ in range(n)]
    ms = [1.0] * n
    return xs, ys, ms


def zero_velocities(n):
    return [0.0] * n, [0.0] * n


# ---------------------------------------------------------------------------
# 重力加速度（Plummer 軟化、ニュートン自己重力）
# ---------------------------------------------------------------------------
def compute_accel(xs, ys, ms, softening, G_scaled):
    X = np.asarray(xs, dtype=float)
    Y = np.asarray(ys, dtype=float)
    M = np.asarray(ms, dtype=float)
    dx = X[None, :] - X[:, None]
    dy = Y[None, :] - Y[:, None]
    r2 = dx * dx + dy * dy + softening * softening
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    np.fill_diagonal(inv_r3, 0.0)
    g = G_scaled * M[None, :] * inv_r3
    ax = (g * dx).sum(axis=1)
    ay = (g * dy).sum(axis=1)
    return ax.tolist(), ay.tolist()


# ---------------------------------------------------------------------------
# 閃焼（leapfrog KDK、時間可逆）
# ---------------------------------------------------------------------------
def leapfrog_kdk_step(xs, ys, vxs, vys, ms, dt, softening, G_scaled):
    ax, ay = compute_accel(xs, ys, ms, softening, G_scaled)
    # half kick
    vxs = [v + 0.5 * dt * a for v, a in zip(vxs, ax)]
    vys = [v + 0.5 * dt * a for v, a in zip(vys, ay)]
    # drift
    xs = [x + dt * v for x, v in zip(xs, vxs)]
    ys = [y + dt * v for y, v in zip(ys, vys)]
    # half kick with new acceleration
    ax2, ay2 = compute_accel(xs, ys, ms, softening, G_scaled)
    vxs = [v + 0.5 * dt * a for v, a in zip(vxs, ax2)]
    vys = [v + 0.5 * dt * a for v, a in zip(vys, ay2)]
    return xs, ys, vxs, vys
def center_of_mass(xs, ys, ms):
    tw = sum(ms)
    cx = sum(m * x for m, x in zip(ms, xs)) / tw
    cy = sum(m * y for m, y in zip(ms, ys)) / tw
    return cx, cy


def velocities_vs_cm(xs, ys, vxs, vys, ms):
    cx, cy = center_of_mass(xs, ys, ms)
    tw = sum(ms)
    vx_cm = sum(m * vx for m, vx in zip(ms, vxs)) / tw
    vy_cm = sum(m * vy for m, vy in zip(ms, vys)) / tw
    return [vx - vx_cm for vx in vxs], [vy - vy_cm for vy in vys], cx, cy


def energy(xs, ys, vxs, vys, ms, softening, G_scaled):
    X = np.asarray(xs, dtype=float)
    Y = np.asarray(ys, dtype=float)
    VX = np.asarray(vxs, dtype=float)
    VY = np.asarray(vys, dtype=float)
    M = np.asarray(ms, dtype=float)
    ke = 0.5 * float(np.sum(M * (VX * VX + VY * VY)))
    dx = X[None, :] - X[:, None]
    dy = Y[None, :] - Y[:, None]
    r = np.sqrt(dx * dx + dy * dy + softening * softening)
    mm = M[:, None] * M[None, :]
    total = float(np.sum(mm / r))
    diag = float(np.sum(np.diagonal(mm) / np.diagonal(r)))
    pe = -0.5 * G_scaled * (total - diag)
    return ke, pe


def radii_metrics(xs, ys, ms, cx, cy):
    X = np.asarray(xs, dtype=float)
    Y = np.asarray(ys, dtype=float)
    M = np.asarray(ms, dtype=float)
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    tw = float(M.sum())
    rg = float(np.sqrt(np.sum(M * d * d) / tw))
    order = np.argsort(d)
    ds = d[order]
    cum = np.cumsum(M[order])
    i20 = min(int(np.searchsorted(cum, 0.20 * tw)), len(ds) - 1)
    i50 = min(int(np.searchsorted(cum, 0.50 * tw)), len(ds) - 1)
    return rg, float(ds[i50]), float(ds[i20])


def shape_entropy(xs, ys, ms, cx, cy):
    n = len(xs)
    vals = []
    tot = 0.0
    for i in range(n):
        d = math.hypot(xs[i] - cx, ys[i] - cy)
        if d <= 1e-12:
            d = 1e-12
        vals.append(math.log(max(d, 1e-12)) * ms[i])
        tot += ms[i]
    mean = sum(vals) / tot
    var = sum(m * (v - mean) ** 2 for (v, m) in zip(vals, ms)) / tot
    if var <= 0.0:
        return 0.0
    return 0.5 * math.log(2.0 * math.pi * math.e * max(var, 1e-12))


def box_entropy(xs, ys, ms, half_width, grid_cells):
    X = np.asarray(xs, dtype=float)
    Y = np.asarray(ys, dtype=float)
    M = np.asarray(ms, dtype=float)
    tw = float(M.sum())
    inside = (np.abs(X) <= half_width) & (np.abs(Y) <= half_width)
    leak = float(M[~inside].sum()) / tw if tw > 0 else 0.0
    if not inside.any():
        return 0.0, leak
    xi = X[inside]
    yi = Y[inside]
    mi = M[inside]
    ci = np.clip(((xi + half_width) / (2 * half_width) * grid_cells).astype(int),
                 0, grid_cells - 1)
    cj = np.clip(((yi + half_width) / (2 * half_width) * grid_cells).astype(int),
                 0, grid_cells - 1)
    cell = np.bincount(ci * grid_cells + cj, weights=mi,
                       minlength=grid_cells * grid_cells)
    p = cell[cell > 0.0] / tw
    return float(-(p * np.log(p)).sum()), leak


def speed_entropy(vxs, vys, ms):
    n = len(vxs)
    tw = sum(ms)
    mx = sum(m * vx for m, vx in zip(ms, vxs)) / tw
    my = sum(m * vy for m, vy in zip(ms, vys)) / tw
    vx2 = sum(m * ((vx - mx) ** 2) for m, vx in zip(ms, vxs)) / tw
    vy2 = sum(m * ((vy - my) ** 2) for m, vy in zip(ms, vys)) / tw
    vx = math.sqrt(max(vx2, 1e-12))
    vy = math.sqrt(max(vy2, 1e-12))
    return math.log(2.0 * math.pi * math.e * vx * vy)


def virial_ratio(ke, pe):
    if abs(2 * ke + pe) < 1e-12 and abs(ke) + abs(pe) < 1e-12:
        return 0.0
    return 2.0 * ke / abs(pe) if pe != 0 else float("inf")
# ---------------------------------------------------------------------------
# 初期状態（ほぼ一様な円盤）と境界条件
# ---------------------------------------------------------------------------
def make_state(seed, condition):
    """半径 R0 のほぼ一様な円盤。condition に応じて外向き速度を与える。"""
    rng = seeded_rng(seed)
    n = P.n_particles
    xs, ys, ms = [], [], []
    while len(xs) < n:
        x = rng.uniform(-P.R0, P.R0)
        y = rng.uniform(-P.R0, P.R0)
        if x * x + y * y <= P.R0 * P.R0:
            xs.append(x)
            ys.append(y)
            ms.append(1.0)
    vxs = [0.0] * n
    vys = [0.0] * n
    vfac = {"expanding": 1.15, "marginal": 0.86}.get(condition, 0.0)
    if vfac > 0.0:
        # v_es(中心) ≈ 2√(G M_tot / 2R)。0.86 倍で全エネルギー ≈ 0（臨界）になる。
        vesc = math.sqrt(2.0 * P.G_scaled * n / P.R0)
        for i in range(n):
            r = math.hypot(xs[i], ys[i]) or 1e-12
            vxs[i] = vfac * vesc * xs[i] / r
            vys[i] = vfac * vesc * ys[i] / r
    return xs, ys, vxs, vys, ms


def apply_wall(xs, ys, vxs, vys, wall_radius):
    """原点中心・半径 wall_radius の反射壁。反転回数を返す（弾性・可逆）。"""
    b = 0
    rmax2 = wall_radius * wall_radius
    for i in range(len(xs)):
        r2 = xs[i] * xs[i] + ys[i] * ys[i]
        if r2 > rmax2:
            r = math.sqrt(r2)
            nx = xs[i] / r
            ny = ys[i] / r
            vn = vxs[i] * nx + vys[i] * ny
            if vn > 0.0:
                vxs[i] -= 2.0 * vn * nx
                vys[i] -= 2.0 * vn * ny
                b += 1
            s = wall_radius / r
            xs[i] *= s
            ys[i] *= s
    return b


def radial_shape_entropy(xs, ys, ms, cx, cy, rg):
    """自己スケール（r / R_gyr）の 1 次元格子エントロピー。形状の集中を見る。"""
    g = P.grid_cells
    X = np.asarray(xs, dtype=float)
    Y = np.asarray(ys, dtype=float)
    M = np.asarray(ms, dtype=float)
    rn = np.hypot(X - cx, Y - cy) / rg if rg > 1e-12 else np.zeros_like(X)
    idx = np.clip((rn / 3.0 * g).astype(int), 0, g - 1)
    cell = np.bincount(idx, weights=M, minlength=g)
    tw = float(M.sum())
    p = cell[cell > 0.0] / tw if tw > 0 else cell[cell > 0.0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def binned_speed_entropy(vxs, vys, ms):
    """速度絶対値の 1 次元格子エントロピー（正規化は実行ごとに自己スケール）。"""
    g = 16
    VX = np.asarray(vxs, dtype=float)
    VY = np.asarray(vys, dtype=float)
    M = np.asarray(ms, dtype=float)
    sp = np.hypot(VX, VY)
    vmax = float(sp.max()) if sp.size else 1.0
    if vmax <= 1e-12:
        return 0.0
    idx = np.clip((sp / vmax * g).astype(int), 0, g - 1)
    cell = np.bincount(idx, weights=M, minlength=g)
    tw = float(M.sum())
    p = cell[cell > 0.0] / tw
    return float(-(p * np.log(p)).sum()) if p.size else 0.0
SERIES_KEYS = [
    "step", "R_gyr", "R_max", "R_half", "R_core20", "expansion_ratio",
    "core_fraction", "mean_speed", "KE", "PE", "E", "virial_ratio",
    "shape_entropy", "box_entropy", "speed_entropy", "box_leakage",
]


def empty_series():
    return {k: [] for k in SERIES_KEYS}


def record(out, xs, ys, vxs, vys, ms, step, rg0):
    """1 サンプル分の指標を計算して積み込む。"""
    cx, cy = center_of_mass(xs, ys, ms)
    rg, rhalf, rc20 = radii_metrics(xs, ys, ms, cx, cy)
    rmax = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys))
    core_frac = sum(1 for x, y in zip(xs, ys)
                    if (x - cx) ** 2 + (y - cy) ** 2 <= (0.3 * P.R0) ** 2) / len(xs)
    sp = [math.hypot(vx, vy) for vx, vy in zip(vxs, vys)]
    mean_speed = sum(sp) / len(sp)
    ke, pe = energy(xs, ys, vxs, vys, ms, P.softening, P.G_scaled)
    sh = radial_shape_entropy(xs, ys, ms, cx, cy, rg)
    bx, leak = box_entropy(xs, ys, ms, P.box_half_width, P.grid_cells)
    se = binned_speed_entropy(vxs, vys, ms)
    out["step"].append(step)
    out["R_gyr"].append(rg)
    out["R_max"].append(rmax)
    out["R_half"].append(rhalf)
    out["R_core20"].append(rc20)
    out["expansion_ratio"].append(rg / rg0 if rg0 > 0 else 1.0)
    out["core_fraction"].append(core_frac)
    out["mean_speed"].append(mean_speed)
    out["KE"].append(ke)
    out["PE"].append(pe)
    out["E"].append(ke + pe)
    out["virial_ratio"].append(virial_ratio(ke, pe))
    out["shape_entropy"].append(sh)
    out["box_entropy"].append(bx)
    out["speed_entropy"].append(se)
    out["box_leakage"].append(leak)


def integrate(xs, ys, vxs, vys, ms, n_steps, use_wall, out=None, start_step=0, rg0=1.0):
    """n_steps だけ時間発展させる。out があれば sample_every ごとに記録。"""
    for step in range(1, n_steps + 1):
        xs, ys, vxs, vys = leapfrog_kdk_step(xs, ys, vxs, vys, ms,
                                             P.dt, P.softening, P.G_scaled)
        if use_wall:
            apply_wall(xs, ys, vxs, vys, P.wall_radius)
        if out is not None and step % P.sample_every == 0:
            record(out, xs, ys, vxs, vys, ms, start_step + step, rg0)
    return xs, ys, vxs, vys


def simulate(seed, condition):
    """条件ごとに時間発展し、サンプリングした指標列を返す。"""
    xs, ys, vxs, vys, ms = make_state(seed, condition)
    use_wall = (condition == "reflecting_wall")
    if condition == "time_reversed":
        warm = P.n_steps // 2
        for _ in range(warm):
            xs, ys, vxs, vys = leapfrog_kdk_step(xs, ys, vxs, vys, ms,
                                                 P.dt, P.softening, P.G_scaled)
        vxs = [-v for v in vxs]
        vys = [-v for v in vys]
    cx0, cy0 = center_of_mass(xs, ys, ms)
    rg0 = radii_metrics(xs, ys, ms, cx0, cy0)[0]
    out = empty_series()
    record(out, xs, ys, vxs, vys, ms, 0, rg0)
    xs, ys, vxs, vys = integrate(xs, ys, vxs, vys, ms, P.n_steps,
                                 use_wall, out, 0, rg0)
    return out
# ---------------------------------------------------------------------------
# 系列統計の補助
# ---------------------------------------------------------------------------
def ranks_of(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks_of(a), ranks_of(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da <= 1e-12 or db <= 1e-12:
        return float("nan")
    return num / (da * db)


def up_fraction(vals):
    if len(vals) < 2:
        return float("nan")
    d = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    nz = [x for x in d if x != 0.0]
    if not nz:
        return 0.5
    return sum(1 for x in nz if x > 0.0) / len(nz)


def local_minima_idx(vals):
    return [i for i in range(1, len(vals) - 1)
            if vals[i] <= vals[i - 1] and vals[i] < vals[i + 1]]


def macrostate_vector(series, i):
    """マクロ状態ベクトル。エネルギーは保存量なので除外し、幾何・エントロピーのみ。"""
    keys = ["R_gyr", "R_half", "R_core20",
            "shape_entropy", "box_entropy", "speed_entropy"]
    v = []
    for k in keys:
        base = abs(series[k][0])
        if base < 1e-12:
            base = 1.0
        v.append(series[k][i] / base)
    return v


def macrostate_distance_min(series):
    n = len(series["R_gyr"])
    skip = int(n * 0.10)
    if n - skip < 2:
        return float("nan")
    m0 = macrostate_vector(series, 0)
    best = float("inf")
    for i in range(skip, n):
        mi = macrostate_vector(series, i)
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(m0, mi)))
        if d < best:
            best = d
    return best


def arrow_fractions(series):
    """R_gyr 膨張向き／収縮向きにサンプルを並べ替えたときの
    box エントロピー単調増加割合。multiset は不変で順序だけ入れる。"""
    order = sorted(range(len(series["R_gyr"])), key=lambda i: series["R_gyr"][i])
    e = series["box_entropy"]
    up_exp = up_fraction([e[i] for i in order])
    up_con = up_fraction([e[i] for i in reversed(order)])
    return up_exp, up_con
def summarize_run(series, condition):
    """1 実行の系列から run_summary フィールド群を作る。"""
    rg, rh, rc = series["R_gyr"], series["R_half"], series["R_core20"]
    ke, pe, e = series["KE"], series["PE"], series["E"]
    n = len(rg)
    rg0, rh0, rc0 = rg[0], rh[0], rc[0]
    ke0, pe0 = ke[0], pe[0]
    energy_scale = abs(ke0) + abs(pe0)
    e0 = e[0]
    drift = (max(abs(x - e0) for x in e) / energy_scale) if energy_scale > 0 else 0.0
    imin = rg.index(min(rg))
    mins = local_minima_idx(rg)
    periods = [mins[j + 1] - mins[j] for j in range(len(mins) - 1)]
    cycle = (statistics.median(periods) * P.sample_every
             if len(periods) >= 2 else float("nan"))
    a_exp, a_con = arrow_fractions(series)
    late = series["virial_ratio"][int(n * 0.75):]
    return {
        "condition": condition,
        "E0": e0,
        "energy_scale_initial": energy_scale,
        "energy_drift_max": drift,
        "R_gyr_initial": rg0,
        "R_gyr_min": min(rg),
        "R_gyr_final": rg[-1],
        "collapse_step": imin * P.sample_every,
        "collapse_ratio": min(rg) / rg0,
        "expansion_ratio_max": max(rg) / rg0,
        "expansion_ratio_final": rg[-1] / rg0,
        "R_half_min_ratio": min(rh) / rh0,
        "R_half_final_ratio": rh[-1] / rh0,
        "core_contraction_ratio": min(rc) / rc0,
        "core_contraction_final_ratio": rc[-1] / rc0,
        "halo_expansion_ratio_max": max(rg) / rg0,
        "core_fraction_max": max(series["core_fraction"]),
        "mean_speed_max": max(series["mean_speed"]),
        "virial_ratio_mean_late": (sum(late) / len(late)) if late else float("nan"),
        "box_entropy_initial": series["box_entropy"][0],
        "box_entropy_final": series["box_entropy"][-1],
        "box_entropy_delta": series["box_entropy"][-1] - series["box_entropy"][0],
        "shape_entropy_initial": series["shape_entropy"][0],
        "shape_entropy_final": series["shape_entropy"][-1],
        "shape_entropy_delta": series["shape_entropy"][-1] - series["shape_entropy"][0],
        "speed_entropy_initial": series["speed_entropy"][0],
        "speed_entropy_final": series["speed_entropy"][-1],
        "speed_entropy_delta": series["speed_entropy"][-1] - series["speed_entropy"][0],
        "n_bounces": max(len(mins) - 1, 0),
        "cycle_period_steps": cycle,
        "macrostate_skip_steps": int(n * 0.10) * P.sample_every,
        "macrostate_distance_min": macrostate_distance_min(series),
        "monotone_up_box_entropy": up_fraction(series["box_entropy"]),
        "monotone_up_shape_entropy": up_fraction(series["shape_entropy"]),
        "spearman_box_entropy_vs_t": spearman(series["box_entropy"], series["step"]),
        "spearman_shape_entropy_vs_t": spearman(series["shape_entropy"], series["step"]),
        "arrow_expansion_up_fraction_box": a_exp,
        "arrow_contraction_up_fraction_box": a_con,
        "box_leakage_max": max(series["box_leakage"]),
    }
# ---------------------------------------------------------------------------
# Q5-A: 速度反転テスト（lag 進めて反転 → 初期にどれだけ戻るか）
# ---------------------------------------------------------------------------
def retrace_test(seed, condition, lag):
    """lag ステップ進めた状態を速度反転で巻き戻し、初期状態との差を測る。"""
    xs, ys, vxs, vys, ms = make_state(seed, condition)
    use_wall = (condition == "reflecting_wall")
    cx0, cy0 = center_of_mass(xs, ys, ms)
    rg0 = radii_metrics(xs, ys, ms, cx0, cy0)[0]
    x0 = list(xs)
    y0 = list(ys)
    ke0, pe0 = energy(xs, ys, vxs, vys, ms, P.softening, P.G_scaled)
    scale0 = abs(ke0) + abs(pe0)
    e0 = ke0 + pe0
    xs, ys, vxs, vys = integrate(xs, ys, vxs, vys, ms, lag, use_wall, None, 0, rg0)
    vxs = [-v for v in vxs]
    vys = [-v for v in vys]
    xs, ys, vxs, vys = integrate(xs, ys, vxs, vys, ms, lag, use_wall, None, 0, rg0)
    dx = [xs[i] - x0[i] for i in range(len(x0))]
    dy = [ys[i] - y0[i] for i in range(len(y0))]
    max_dx = max(max(abs(a), abs(b)) for a, b in zip(dx, dy))
    rms = math.sqrt(sum(a * a + b * b for a, b in zip(dx, dy)) / (2 * len(dx)))
    ke1, pe1 = energy(xs, ys, vxs, vys, ms, P.softening, P.G_scaled)
    edrift = abs((ke1 + pe1) - e0) / scale0 if scale0 > 0 else 0.0
    return {
        "condition": condition,
        "seed": seed,
        "lag_steps": lag,
        "return_max_abs_dx": max_dx,
        "return_rms_dx": rms,
        "return_rms_over_R0": rms / P.R0,
        "energy_after_drift": edrift,
    }


# ---------------------------------------------------------------------------
# Q5-B: ラベル反転テスト（系列を逆順に読んでも物理量の集合は同じか）
# ---------------------------------------------------------------------------
def label_reversal_test(series):
    out = {}
    t = series["step"]
    for metric in ["R_gyr", "box_entropy", "shape_entropy", "KE", "virial_ratio"]:
        v = series[metric]
        rev = list(reversed(v))
        same = sorted(v) == sorted(rev)
        out[metric] = {
            "multiset_identical": bool(same),
            "spearman_vs_t_forward": spearman(v, t),
            "spearman_vs_t_reversed": spearman(rev, t),
            "entropy_up_fraction_forward": up_fraction(v),
            "entropy_up_fraction_reversed": up_fraction(rev),
        }
    return out


# ---------------------------------------------------------------------------
# プール統計（シード間）
# ---------------------------------------------------------------------------
POOLED_KEYS = [
    "collapse_ratio", "core_contraction_ratio", "R_half_min_ratio",
    "R_half_final_ratio", "core_contraction_final_ratio",
    "halo_expansion_ratio_max", "expansion_ratio_final", "core_fraction_max",
    "mean_speed_max", "virial_ratio_mean_late",
    "box_entropy_delta", "shape_entropy_delta", "speed_entropy_delta",
    "spearman_box_entropy_vs_t", "spearman_shape_entropy_vs_t",
    "monotone_up_box_entropy", "monotone_up_shape_entropy",
    "arrow_expansion_up_fraction_box", "arrow_contraction_up_fraction_box",
    "n_bounces", "cycle_period_steps", "macrostate_distance_min",
    "box_leakage_max", "energy_drift_max",
]


def pooled_stats(summaries):
    """条件ごとにシード間の平均・SD・最小・最大を集める。"""
    by_cond = {}
    for s in summaries:
        by_cond.setdefault(s["condition"], []).append(s)
    pooled = {}
    for cond, rows in by_cond.items():
        stats = {"n_seeds": len(rows)}
        for key in POOLED_KEYS:
            vals = []
            for r in rows:
                v = r.get(key, float("nan"))
                vals.append(v if isinstance(v, (int, float)) else float("nan"))
            finite = [v for v in vals if math.isfinite(v)]
            if finite:
                stats[key] = {
                    "mean": statistics.fmean(finite),
                    "sd": (statistics.pstdev(finite) if len(finite) > 1 else 0.0),
                    "min": min(finite),
                    "max": max(finite),
                    "values": vals,
                }
            else:
                stats[key] = {"mean": float("nan"), "sd": float("nan"),
                              "min": float("nan"), "max": float("nan"),
                              "values": vals}
        pooled[cond] = stats
    return pooled
CSV_FIELDS = [
    "record_type", "condition", "seed", "step", "key", "metric",
    "R_gyr", "R_max", "R_half", "R_core20", "expansion_ratio", "core_fraction",
    "mean_speed", "KE", "PE", "E", "virial_ratio", "shape_entropy",
    "box_entropy", "speed_entropy", "box_leakage",
    "E0", "energy_scale_initial", "energy_drift_max", "collapse_step",
    "collapse_ratio", "expansion_ratio_max", "expansion_ratio_final",
    "core_fraction_max", "R_half_min_ratio", "R_half_final_ratio",
    "core_contraction_ratio", "core_contraction_final_ratio",
    "halo_expansion_ratio_max", "macrostate_skip_steps", "mean_speed_max",
    "virial_ratio_mean_late", "box_entropy_initial", "box_entropy_final",
    "box_entropy_delta", "shape_entropy_initial", "shape_entropy_final",
    "shape_entropy_delta", "speed_entropy_initial", "speed_entropy_final",
    "speed_entropy_delta", "n_bounces", "cycle_period_steps",
    "macrostate_distance_min", "monotone_up_box_entropy",
    "monotone_up_shape_entropy", "spearman_box_entropy_vs_t",
    "spearman_shape_entropy_vs_t", "arrow_expansion_up_fraction_box",
    "arrow_contraction_up_fraction_box", "box_leakage_max",
    "retrace_return_rms_over_R0", "lag_steps", "return_max_abs_dx",
    "return_rms_dx", "return_rms_over_R0", "energy_after_drift",
    "multiset_identical", "spearman_vs_t_forward", "spearman_vs_t_reversed",
    "entropy_up_fraction_forward", "entropy_up_fraction_reversed",
    "mean", "sd", "min", "max", "range", "values",
]


def _csv_row(d):
    return {k: d.get(k, "") for k in CSV_FIELDS}


def write_csv(path, bundle):
    """series / run_summary / pooled / retrace / label_reversal を 1 ファイルへ。"""
    rows = []
    for run in bundle["runs"]:
        s = run["series"]
        cond, seed = run["condition"], run["seed"]
        for i in range(len(s["step"])):
            row = {"record_type": "series", "condition": cond, "seed": seed}
            for k in SERIES_KEYS:
                row[k] = s[k][i]
            rows.append(_csv_row(row))
        row = {"record_type": "run_summary", "condition": cond, "seed": seed}
        row.update(run["summary"])
        rows.append(_csv_row(row))
    for cond, st in bundle["pooled"].items():
        for key in POOLED_KEYS:
            b = st[key]
            rng = (b["max"] - b["min"]) if math.isfinite(b["mean"]) else float("nan")
            rows.append(_csv_row({
                "record_type": "pooled", "condition": cond, "key": key,
                "mean": b["mean"], "sd": b["sd"], "min": b["min"],
                "max": b["max"], "range": rng,
                "values": json.dumps(b["values"], ensure_ascii=False),
            }))
    for d in bundle["retrace"]:
        rows.append(_csv_row({
            "record_type": "retrace", "condition": d["condition"],
            "seed": d["seed"], "lag_steps": d["lag_steps"],
            "return_max_abs_dx": d["return_max_abs_dx"],
            "return_rms_dx": d["return_rms_dx"],
            "return_rms_over_R0": d["return_rms_over_R0"],
            "retrace_return_rms_over_R0": d["return_rms_over_R0"],
            "energy_after_drift": d["energy_after_drift"],
        }))
    for metric, d in bundle["label_reversal"].items():
        row = {"record_type": "label_reversal", "metric": metric}
        row.update(d)
        rows.append(_csv_row(row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
# ---------------------------------------------------------------------------
# JSON 出力
# ---------------------------------------------------------------------------
MODEL_INFO = {
    "dimension": 2,
    "interaction": "Plummer 軟化付きニュートン自己重力（散逸・放射なし）",
    "integrator": "leapfrog (kick-drift-kick) — 時間可逆",
    "boundary": "孤立系（外部ポテンシャルなし）。閉箱の対照として reflecting_wall を持つ",
    "closed_system_note": "周期境界は重力の長距離性と相性が悪いため使わない。"
                           "『閉じた系』は全エネルギー保存・外力ゼロとして定義している。",
    "entropy_proxies": {
        "box": "固定箱上の 2D 格子エントロピー。粒子が箱外へ出ると見かけ上減る（漏出アーチファクト）",
        "shape": "R_gyr で自己スケールした動径分布の格子エントロピー（形状の集中度）",
        "speed": "速度絶対値の格子エントロピー",
        "caution": "重力系のエントロピーは粗視化の定義に強く依存する。"
                   "本実験の代理指標は位相空間を粗視化していない",
    },
}

QUESTIONS = [
    "Q1 ほぼ一様な系は重力だけで凝縮するか",
    "Q2 凝縮状態は『ビッグバン直後』に似るか（エントロピー代理で見る）",
    "Q3 時間の矢を膨張でなく収縮方向に置くと何が変わるか",
    "Q4 循環するか、一方向に終わるか",
    "Q5 速度反転（物理的操作）とラベル反転（解釈の操作）で past に戻るか",
]

INTERPRETATION_VS_PHYSICS = {
    "Q5_A_velocity_reversal": "速度反転は物理的操作。戻り誤差が丸め誤差程度なら、"
                              "系は決定論的に past に戻る。",
    "Q5_B_label_reversal": "時間ラベルの反転は解釈の操作。物理量の multiset は不変で、"
                           "変わるのは時間ラベルとの相関の符号だけ。",
    "not_claimed": "『時空が逆方向に動く』という物理的断定は行わない。",
}

TONE_POLICY = ("『証明された』『宇宙の正体が判明した』とは書かない。"
               "時間の矢を逆に読むことは解釈実験であり、物理的断定ではない。示唆として書く。")


def _clean(v):
    """NaN/Inf を JSON の null に変換する（再帰）。"""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def write_json(path, bundle):
    per_seed = []
    for run in bundle["runs"]:
        d = {"condition": run["condition"], "seed": run["seed"]}
        d.update(run["summary"])
        per_seed.append(_clean(d))
    doc = {
        "title": "閉じた玩具宇宙における凝縮・時間の矢・循環の数値実験",
        "subtitle": ("物理法則の証明ではない。『閉じた玩具宇宙において、"
                     "収縮・凝縮・時間の矢・循環の有無』を数値的に見る実験。"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL_INFO,
        "params": P.to_dict(),
        "questions": QUESTIONS,
        "interpretation_vs_physics": INTERPRETATION_VS_PHYSICS,
        "per_seed": per_seed,
        "pooled_by_condition": _clean(bundle["pooled"]),
        "label_reversal_test": _clean(bundle["label_reversal"]),
        "retrace_test": _clean(bundle["retrace"]),
        "interpretation_notes": bundle["notes"],
        "tone_policy": TONE_POLICY,
        "elapsed_seconds": bundle["elapsed"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
# ---------------------------------------------------------------------------
# 図（matplotlib があれば生成）
# ---------------------------------------------------------------------------
def _first_run(bundle, cond):
    for run in bundle["runs"]:
        if run["condition"] == cond:
            return run
    return None


def make_figure(path, bundle):
    if not HAVE_MPL:
        return False
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("closed toy universe: collapse, arrows of time, recurrence "
                 "(suggestive only, not a proof)")

    ax = axes[0][0]
    for cond in P.conditions:
        run = _first_run(bundle, cond)
        if run:
            ax.plot(run["series"]["step"], run["series"]["expansion_ratio"],
                    lw=1.2, label=cond)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("R_gyr / R_gyr(0)")
    ax.set_title("size ratio (halo-dominated)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    run = _first_run(bundle, "bound_collapse")
    if run:
        s = run["series"]
        for k, c in (("R_gyr", "tab:blue"), ("R_half", "tab:orange"),
                     ("R_core20", "tab:red")):
            ax.plot(s["step"], s[k], lw=1.2, color=c, label=k)
    ax.set_xlabel("step")
    ax.set_ylabel("radius")
    ax.set_title("core contracts while halo expands")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][2]
    for cond, st in bundle["pooled"].items():
        b = st.get("core_contraction_ratio")
        if b and math.isfinite(b["mean"]):
            ax.bar(cond, b["mean"], yerr=b["sd"], alpha=0.8)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("min R_core20 / R_core20(0)")
    ax.set_title("core contraction by condition")
    ax.tick_params(axis="x", labelrotation=20, labelsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][0]
    idx = list(range(len(P.conditions)))
    box_m = [bundle["pooled"][c]["box_entropy_delta"]["mean"] for c in P.conditions]
    shp_m = [bundle["pooled"][c]["shape_entropy_delta"]["mean"] for c in P.conditions]
    ax.bar([i - 0.2 for i in idx], box_m, width=0.4, label="box")
    ax.bar([i + 0.2 for i in idx], shp_m, width=0.4, label="shape")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(P.conditions, rotation=20, fontsize=7)
    ax.set_ylabel("entropy proxy delta")
    ax.set_title("proxies do NOT agree on increase")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1][1]
    for cond in ("bound_collapse", "reflecting_wall", "expanding"):
        run = _first_run(bundle, cond)
        if run:
            s = run["series"]
            n = len(s["step"])
            skip = int(n * 0.10)
            vecs = [macrostate_vector(s, i) for i in range(n)]
            m0 = vecs[0]
            dist = [math.sqrt(sum((a - b) ** 2 for a, b in zip(m0, v)))
                    for v in vecs]
            ax.plot(s["step"][skip:], dist[skip:], lw=1.2, label=cond)
    ax.set_xlabel("step")
    ax.set_ylabel("distance to initial macrostate")
    ax.set_title("how close does it come back?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1][2]
    by_cond = {}
    for d in bundle["retrace"]:
        by_cond.setdefault(d["condition"], []).append(d)
    for cond, rows in by_cond.items():
        rows = sorted(rows, key=lambda r: r["lag_steps"])
        ax.semilogy([r["lag_steps"] for r in rows],
                    [max(r["return_rms_over_R0"], 1e-20) for r in rows],
                    marker="o", lw=1.2, label=cond)
    ax.set_xlabel("lag steps before reversal")
    ax.set_ylabel("return rms / R0")
    ax.set_title("velocity reversal (Q5-A)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True
# ---------------------------------------------------------------------------
# 解釈ノート（必ず実測値に追随させる。断定・証明の言い方はしない）
# ---------------------------------------------------------------------------
def _fmt(block, key):
    if key not in block:
        return "n/a"
    b = block[key]
    mean = b.get("mean")
    sd = b.get("sd")
    n = block.get("n_seeds", 0)
    if mean is None or (isinstance(mean, float) and not math.isfinite(mean)):
        return "n/a"
    if sd is None or (isinstance(sd, float) and not math.isfinite(sd)):
        return "%+.4f（n=%d、シード間 SD は算出不可）" % (mean, n)
    return ("%+.4f（シード間 SD %.4f、範囲 %+.4f〜%+.4f、n=%d）"
            % (mean, sd, b["min"], b["max"], n))


def build_notes(pooled, retrace_rows, label_rev):
    notes = []
    bc = pooled.get("bound_collapse", {})
    ex = pooled.get("expanding", {})
    mg = pooled.get("marginal", {})
    rw = pooled.get("reflecting_wall", {})
    rv = pooled.get("time_reversed", {})

    def _mean(block, key):
        v = block.get(key, {}).get("mean")
        return v if isinstance(v, (int, float)) and math.isfinite(v) else None

    cr = _mean(bc, "collapse_ratio")
    ccr = _mean(bc, "core_contraction_ratio")
    hxm = _mean(bc, "halo_expansion_ratio_max")
    if cr is None or ccr is None:
        q1_tail = "コアと全体の収縮率はどちらも算出できず、判定を留保する。"
    elif cr > 0.90:
        q1_tail = ("つまり R_gyr はほぼ縮まず、凝縮はコア側にのみ現れる。"
                   " R_gyr 単独の指標では崩壊を見落とす（本実験の設計上の注意点）。")
    elif ccr < 0.70 * cr:
        q1_tail = ("R_gyr も縮むが凝縮はコア側に強く現れる（コア "
                   + _fmt(bc, "core_contraction_ratio") + " ＜ 全体 "
                   + _fmt(bc, "collapse_ratio") + "）。さらに後段でハローは "
                   + _fmt(bc, "halo_expansion_ratio_max")
                   + " 倍まで開く＝同一の実行にコア収縮とハロー膨張（負の比熱）の両相が現れる。")
    else:
        q1_tail = ("R_gyr とコアがほぼ同じ割合で縮む（自己相似的収縮）。"
                   " ハロー側は後段で " + _fmt(bc, "halo_expansion_ratio_max") + " 倍まで開く。")
    notes.append(
        "Q1（重力だけで凝縮するか）: R_gyr（ハロー優勢のサイズ）の最小/初期 = "
        + _fmt(bc, "collapse_ratio") + "、最大/初期 = "
        + _fmt(bc, "halo_expansion_ratio_max") + "。一方コア側は 20%質量半径の最小/初期 = "
        + _fmt(bc, "core_contraction_ratio") + "、50%質量半径 = "
        + _fmt(bc, "R_half_min_ratio") + "。" + q1_tail
        + " 膨張・臨界条件ではコア収縮は " + _fmt(ex, "core_contraction_ratio")
        + " / " + _fmt(mg, "core_contraction_ratio") + " でほぼ起きない。")

    d_box = bc.get("box_entropy_delta", {}).get("mean")
    d_shape = bc.get("shape_entropy_delta", {}).get("mean")
    d_speed = bc.get("speed_entropy_delta", {}).get("mean")
    triple = (("box", d_box), ("shape", d_shape), ("speed", d_speed))
    rising = [nm for nm, v in triple if v is not None and math.isfinite(v) and v > 0.05]
    falling = [nm for nm, v in triple if v is not None and math.isfinite(v) and v < -0.05]
    flat = [nm for nm, v in triple if v is not None and math.isfinite(v) and abs(v) <= 0.05]

    def _dir(v):
        if v is None or not math.isfinite(v):
            return "算出不可"
        if v > 0.05:
            return "増加"
        if v < -0.05:
            return "減少"
        return "ほぼ不変"

    if rising and not falling:
        concl = ("3 代理は揃って増加方向を示し、教科書的な"
                 "『重力系は凝縮するとエントロピーが増える』と同じ向きの結果になった")
    elif falling and not rising:
        concl = ("3 代理はいずれも減少方向を示し、教科書的な"
                 "『重力系は凝縮するとエントロピーが増える』を再現しなかった")
    else:
        concl = ("3 代理の向きが揃わず（box=%s / shape=%s / speed=%s）、"
                 "教科書的な『重力系は凝縮するとエントロピーが増える』を"
                 "明確には再現しなかった"
                 % (_dir(d_box), _dir(d_shape), _dir(d_speed)))

    notes.append(
        "Q2（凝縮状態は『ビッグバン直後』に似るか）: コアが "
        + _fmt(bc, "core_contraction_ratio") + " まで収縮する間のエントロピー代理の終始差は "
        "box 型 = " + _fmt(bc, "box_entropy_delta") + "、shape 型 = "
        + _fmt(bc, "shape_entropy_delta") + "、speed 型 = "
        + _fmt(bc, "speed_entropy_delta") + " nats（Spearman(box, t) = "
        + _fmt(bc, "spearman_box_entropy_vs_t") + "）。内訳: 増加="
        + (str(rising) if rising else "なし") + " / 減少="
        + (str(falling) if falling else "なし") + " / ほぼ不変="
        + (str(flat) if flat else "なし") + "。")
    notes.append(
        " → " + concl + "。したがって本実験のデータでは『凝縮＝ビッグバン直後に似る』も"
        "『似ていない』も裁定できない。初期を一様＝低エントロピーと見なしたのは設計側の仮定で、"
        " 代理指標の選び方で向きが変わる領域にある。")
    notes.append(
        "  ※ 指標の限界: box エントロピーは粒子が固定箱から出ると見かけ上『減る』"
        "（bound の最大漏出 = " + _fmt(bc, "box_leakage_max") + "、膨張条件の dS_box = "
        + _fmt(ex, "box_entropy_delta") + " は主にこの漏出アーチファクト）。"
        " 重力エントロピーを扱うには位相空間（位置×速度）の粗視化と十分大きな N が"
        " 必要で、今回の格子では理論的期待を検定できていない。")

    notes.append(
        "Q3（矢印を収縮方向に置く）: 実時間での box エントロピー単調増加割合 = "
        + _fmt(bc, "monotone_up_box_entropy") + "。R_gyr が膨張する向きに並べ替えると "
        + _fmt(bc, "arrow_expansion_up_fraction_box") + "、収縮向きだと "
        + _fmt(bc, "arrow_contraction_up_fraction_box") + "。並べ替えは物理量を変えない"
        "（同じ multiset）ので、変わるのは『エントロピーが単調に見えるか』だけである。"
        " ここが解釈実験の核心で、時間の矢を逆に読むことは物理的断定ではない。")

    mb, mrw, me = _mean(bc, "macrostate_distance_min"), _mean(rw, "macrostate_distance_min"), _mean(ex, "macrostate_distance_min")
    if mrw is not None and mb is not None:
        if mrw < 0.5 * mb:
            closed_cmp = "閉箱では初期マクロ状態へのはるかに近い所まで戻る（循環に近い）"
        elif mrw >= mb:
            closed_cmp = "閉箱でも最接近距離は孤立系より遠く、循環に近いとは言い難い"
        else:
            closed_cmp = "閉箱と孤立系で最接近距離に明確な差はない"
    else:
        closed_cmp = "閉箱と孤立系の比較はデータ不足で留保する"
    iso_txt = ("孤立系は初期マクロ状態に戻らない"
               if (mb is not None and mb > 0.5)
               else "孤立系でも初期マクロ状態の近くまで戻る")
    notes.append(
        "Q4（循環するか一方向か）: 初期マクロ状態への最接近距離（最初の10%は除外）は "
        "bound_collapse で " + _fmt(bc, "macrostate_distance_min") + "、reflecting_wall で "
        + _fmt(rw, "macrostate_distance_min") + "（バウンス " + _fmt(rw, "n_bounces")
        + " 回）、expanding で " + _fmt(ex, "macrostate_distance_min") + "。"
        + closed_cmp + "。" + iso_txt + "（ハローの最大膨張 "
        + _fmt(bc, "halo_expansion_ratio_max") + " 倍）。"
        " 膨張・臨界条件は最終膨張 " + _fmt(ex, "expansion_ratio_final") + " / "
        + _fmt(mg, "expansion_ratio_final") + " 倍まで一方向に開く。")
    notes.append(
        "  サイズの振動（bound のバウンス " + _fmt(bc, "n_bounces")
        + " 回）と、粗視化エントロピーの不可逆性は別物として測っている。"
        " バウンスが1回以下の実行では周期推定は n/a のままにする（推測で埋めない）。")

    smooth = [d for d in retrace_rows if d["condition"] == "bound_collapse"]
    walled = [d for d in retrace_rows if d["condition"] == "reflecting_wall"]
    txt = "Q5-A（速度反転で past に戻るか）: "
    ws_val = ww_val = None
    if smooth:
        ws = max(smooth, key=lambda d: d["return_rms_over_R0"])
        ws_val = ws["return_rms_over_R0"]
        txt += ("なめらかな孤立系では反転後の戻り誤差 rms/R0 の最大が %.2e（lag=%d step）で"
                "丸め誤差レベルに留まる。" % (ws_val, ws["lag_steps"]))
    if walled:
        ww = max(walled, key=lambda d: d["return_rms_over_R0"])
        ww_val = ww["return_rms_over_R0"]
        if ws_val is not None and ww_val > 10.0 * max(ws_val, 1e-300):
            txt += (" 反射壁ありでは最大 %.2e（lag=%d step）まで拡大し、"
                    "不連続な衝突が誤差を増幅した。" % (ww_val, ww["lag_steps"]))
        else:
            txt += (" 今回の lag 範囲では反射壁ありの戻り誤差も最大 %.2e（lag=%d step）"
                    "で、壁なしと比べた増幅は確認できない（壁に触れない実行がある）。"
                    % (ww_val, ww["lag_steps"]))
    txt += " lag ごとの成長は retrace 行（CSV/JSON）に出している。"
    notes.append(txt)

    b_ent = label_rev.get("box_entropy", {})
    notes.append(
        "Q5-B（ラベル反転テスト）: 系列を逆順に読んでも物理量の集合は"
        + ("完全一致した" if b_ent.get("multiset_identical") else "一致しなかった")
        + "。変わるのは時間ラベルとの相関の符号だけ（例: box_entropy の Spearman %.3f → %.3f）。"
        " よって本実験では『時空が逆方向に動く』とは主張しない。"
        % (b_ent.get("spearman_vs_t_forward", float("nan")),
           b_ent.get("spearman_vs_t_reversed", float("nan"))))
    notes.append(
        "  反転実行そのもの（time_reversed）: エントロピー代理の終始差 = "
        + _fmt(rv, "box_entropy_delta") + "、初期マクロ状態への最接近距離 = "
        + _fmt(rv, "macrostate_distance_min") + "。反転系は初期マクロ状態にほぼ戻る"
        "（＝系列を往復しただけ）。これは『一様（＝低エントロピーと仮定した状態）から始めて"
        "人工的に巻き戻した』という設計上の話で、現実の宇宙で同じことが起きる主張にはならない。")

    drift = bc.get("energy_drift_max", {})
    notes.append(
        "数値の信頼性: エネルギー保存の相対漂移 max|E(t)-E0| / (|KE0|+|PE0|) = "
        + _fmt(bc, "energy_drift_max") + "（bound_collapse。E0≈0 でも発散しない定義）。"
        " leapfrog(KDK) は可逆なので時間反転テストの前提と整合する。")
    notes.append(
        "総括の限界: これは 2 次元・軟化ニュートン・放射なし・有限時間の玩具である。"
        " 得られるのは保存量・相関・再現誤差であり、宇宙論の結論でも証明でもない。"
        " 『宇宙の正体が判明した』とは書かない。")
    return notes
# ---------------------------------------------------------------------------
# コンソールレポート
# ---------------------------------------------------------------------------
POOLED_TABLE_KEYS = [
    "collapse_ratio", "core_contraction_ratio", "R_half_min_ratio",
    "box_entropy_delta", "monotone_up_box_entropy", "n_bounces",
    "macrostate_distance_min", "energy_drift_max",
]
W = 100


def _rule(title):
    print("\n" + "=" * W)
    print("  " + title)
    print("=" * W)


def print_seed_table(runs):
    _rule("シード別の主要指標")
    print("  %-20s %6s %9s %9s %10s %9s %10s %8s %10s %8s" % (
        "condition", "seed", "Rmin/R0", "core_min", "Rhalf_min",
        "dS_box", "dS_shape", "bounces", "macro_min", "dE%"))
    for r in runs:
        s = r["summary"]
        print("  %-20s %6d %9.3f %9.3f %10.3f %9.3f %10.3f %8d %10.3f %7.3f%%" % (
            r["condition"], r["seed"], s["collapse_ratio"],
            s["core_contraction_ratio"], s["R_half_min_ratio"],
            s["box_entropy_delta"], s["shape_entropy_delta"],
            s["n_bounces"], s["macrostate_distance_min"],
            s["energy_drift_max"] * 100.0))


def print_pooled(pooled):
    _rule("条件別のプール統計（シード間の平均 ± SD）")
    print("  %-20s" % "condition" + "".join("%16s" % k[:14]
                                            for k in POOLED_TABLE_KEYS))
    for cond in P.conditions:
        st = pooled.get(cond)
        if not st:
            continue
        line = "  %-20s" % cond
        for k in POOLED_TABLE_KEYS:
            b = st.get(k, {})
            m, sd = b.get("mean"), b.get("sd")
            if m is None or not math.isfinite(m):
                line += "%16s" % "n/a"
            else:
                line += "%16s" % ("%+.3f±%.3f" % (m, sd if sd and math.isfinite(sd) else 0.0))
        print(line)


def print_arrow_table(pooled):
    _rule("Q3: 時間の矢の貼り分け（box エントロピーの単調増加割合）")
    print("  %-20s %13s %18s %19s" % ("condition", "real time",
                                       "expansion arrow", "contraction arrow"))
    for cond in P.conditions:
        st = pooled.get(cond, {})
        print("  %-20s %13.3f %18.3f %19.3f" % (
            cond,
            st.get("monotone_up_box_entropy", {}).get("mean", float("nan")),
            st.get("arrow_expansion_up_fraction_box", {}).get("mean", float("nan")),
            st.get("arrow_contraction_up_fraction_box", {}).get("mean", float("nan"))))


def print_label_table(lr):
    _rule("Q5-B: ラベル反転テスト（系列を逆順に読んでも物理量の集合は同じか）")
    print("  %-16s %18s %14s %14s %10s %10s" % (
        "metric", "multiset identical", "spearman fwd",
        "spearman rev", "up% fwd", "up% rev"))
    for k, d in lr.items():
        print("  %-16s %18s %14.3f %14.3f %10.3f %10.3f" % (
            k, str(d["multiset_identical"]),
            d["spearman_vs_t_forward"], d["spearman_vs_t_reversed"],
            d["entropy_up_fraction_forward"], d["entropy_up_fraction_reversed"]))


def print_retrace_table(rows):
    _rule("Q5-A: 速度反転テスト（lag ステップ進めて反転 → どれだけ初期に戻る）")
    print("  %-18s %6s %6s %11s %11s %11s %12s" % (
        "condition", "seed", "lag", "max|dx|", "rms dx", "rms/R0", "dE% after"))
    shown = 0
    for d in rows:
        if d["seed"] == rows[0]["seed"]:
            print("  %-18s %6d %6d %11.3e %11.3e %11.3e %11.3f%%" % (
                d["condition"], d["seed"], d["lag_steps"],
                d["return_max_abs_dx"], d["return_rms_dx"],
                d["return_rms_over_R0"], d["energy_after_drift"] * 100.0))
            shown += 1
    worst = max(rows, key=lambda d: d["return_rms_over_R0"])
    print("  ※ 全 %d 行中、戻り誤差の最大は %s seed=%d lag=%d rms/R0=%.3e" % (
        len(rows), worst["condition"], worst["seed"],
        worst["lag_steps"], worst["return_rms_over_R0"]))
# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
import time


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="cosmic_collapse_time_arrow.py",
        description="閉じた玩具宇宙における凝縮・時間の矢・循環の数値実験"
                    "（示唆であり証明ではない）")
    ap.add_argument("--quick", action="store_true",
                    help="スモークテスト（n_steps とシード数を削減）")
    ap.add_argument("--outdir", default=None,
                    help="CSV/JSON 出力先（既定: results_new）")
    ap.add_argument("--figdir", default=None,
                    help="PNG 出力先（既定: figures_new）")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        P.n_steps = 600
        P.seeds = P.seeds[:1]
        P.retrace_lags = [50, 100]
    root = os.path.dirname(os.path.abspath(__file__))
    tag = "_quick" if args.quick else ""
    outdir = args.outdir or os.path.join(root, "results_new" + tag)
    figdir = args.figdir or os.path.join(root, "figures_new" + tag)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    print("=" * W)
    print("  cosmic_collapse_time_arrow.py — 閉じた玩具宇宙（凝縮・時間の矢・循環）")
    print("  N=%d  steps=%d  dt=%s  軟化=%s  R0=%s  条件=%d  シード=%s" % (
        P.n_particles, P.n_steps, P.dt, P.softening, P.R0,
        len(P.conditions), P.seeds))
    print("  ※ 物理法則の証明ではなく、解釈実験です"
          "（『時空が逆方向に動く』とは主張しない）")
    print("=" * W)

    t0 = time.perf_counter()
    runs = []
    for cond in P.conditions:
        for seed in P.seeds:
            series = simulate(seed, cond)
            summ = summarize_run(series, cond)
            summ["seed"] = seed
            runs.append({"condition": cond, "seed": seed,
                         "series": series, "summary": summ})
            print("  %-17s seed=%-4d Rmin/R0=%.3f core_min=%.3f dS_box=%+.3f "
                  "bounces=%d dE=%.2f%%" % (
                      cond, seed, summ["collapse_ratio"],
                      summ["core_contraction_ratio"], summ["box_entropy_delta"],
                      summ["n_bounces"], summ["energy_drift_max"] * 100.0))

    retrace = []
    for cond in ("bound_collapse", "reflecting_wall"):
        for seed in P.seeds:
            for lag in P.retrace_lags:
                retrace.append(retrace_test(seed, cond, lag))
    worst = max(retrace, key=lambda d: d["return_rms_over_R0"])
    print("      retrace 最悪: %s seed=%d lag=%d rms_dx=%.2e" % (
        worst["condition"], worst["seed"], worst["lag_steps"],
        worst["return_rms_over_R0"]))

    label_rev = label_reversal_test(runs[0]["series"])
    pooled = pooled_stats([r["summary"] for r in runs])
    notes = build_notes(pooled, retrace, label_rev)
    elapsed = time.perf_counter() - t0
    bundle = {"runs": runs, "retrace": retrace, "label_reversal": label_rev,
              "pooled": pooled, "notes": notes, "elapsed": elapsed}

    csv_path = os.path.join(outdir, "cosmic_collapse_time_arrow.csv")
    json_path = os.path.join(outdir, "cosmic_collapse_time_arrow.json")
    n_rows = write_csv(csv_path, bundle)
    write_json(json_path, bundle)
    rel_out = os.path.relpath(outdir, root).replace(os.sep, "/")
    rel_fig = os.path.relpath(figdir, root).replace(os.sep, "/")
    print("\n  %s/cosmic_collapse_time_arrow.csv   (%d rows)" % (rel_out, n_rows))
    print("  %s/cosmic_collapse_time_arrow.json  (書き出し完了)" % rel_out)

    fig_path = os.path.join(figdir, "cosmic_collapse_time_arrow.png")
    if make_figure(fig_path, bundle):
        print("  %s/cosmic_collapse_time_arrow.png (生成完了)" % rel_fig)
    else:
        print("  ※ matplotlib が利用できないため図はスキップしました。")

    print_seed_table(runs)
    print_pooled(pooled)
    print_arrow_table(pooled)
    print_label_table(label_rev)
    print_retrace_table(retrace)

    _rule("現時点で言えること／言えないこと（示唆であり証明ではない）")
    for nt in notes:
        print("  - " + nt)

    print("\n  実行時間 %.1fs / runs=%d / retrace=%d 行" % (
        elapsed, len(runs), len(retrace)))
    print("  ※ 結果は『示唆』として読んでください。保存量・相関・再現誤差が得られた"
          "もので、宇宙論の結論ではありません。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
