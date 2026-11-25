#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rotation_minimal.py — 回転対称ポテンシャルにおける逆πノイズの小規模対照（玩具・示唆）

物理法則や最適化の一般定理の「証明」ではありません。数値実験の示唆どまりです。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# 既定パラメータ（既存図のヘッダ・pyc 復元より推定）
# ---------------------------------------------------------------------------
A_AMPLITUDE = 10.0
BOUND = 5.0
T0 = 10.0
COOLING = 0.997
N_STEPS = 3000
N_TRIALS = 100
TOL = 0.1
STEP_SCALE = 0.35
SEED = 42
NUM_DIGITS = 10_000
Z_95 = 1.959963984540054
ALPHA = 0.05
MULTISEED = [42, 7, 13, 29, 101]

NOISE_TYPES = [
    "pi",
    "inverse_pi",
    "inverse_pi_mean_matched",
    "uniform_nonzero",
    "pi_nonzero_resampled",
]
NOISE_LABELS = {
    "pi": "pi (0-9)",
    "inverse_pi": "inverse pi (1-10)",
    "inverse_pi_mean_matched": "inverse pi (mean-matched)",
    "uniform_nonzero": "uniform 1-10",
    "pi_nonzero_resampled": "pi non-zero resampled",
}
NOISE_COLORS = {
    "pi": "#00ccff",
    "inverse_pi": "#ff8800",
    "inverse_pi_mean_matched": "#ffdd00",
    "uniform_nonzero": "#66ff99",
    "pi_nonzero_resampled": "#cc66ff",
}
COMPARISON_PAIRS = [
    ("pi", "inverse_pi", "pi (0-9) vs inverse pi (1-10)"),
    ("pi", "inverse_pi_mean_matched", "pi (0-9) vs inverse pi (mean-matched)"),
]

MODULE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODULE_DIR / "results"
FIGURES_DIR = MODULE_DIR / "figures"

DARK_BG = "#0a0a0a"
DARK_GRID = "#2b2b2b"
DARK_TEXT = "#e6e6e6"
DARK_SUBTLE = "#8b8b8b"

_PI_1000 = (
    "141592653589793238462643383279502884197169399375105820974944592307816406286"
    "208998628034825342117067982148086513282306647093844609550582231725359408128"
    "481117450284102701938521105559644622948954930381964428810975665933446128475"
    "648233786783165271201909145648566923460348610454326648213393607260249141273"
    "724587006606315588174881520920962829254091715364367892590360011330530548820"
    "466521384146951941511609433057270365759591953092186117381932611793105118548"
    "074462379962749567351885752724891227938183011949129833673362440656643086021"
    "394946395224737190702179860943702770539217176293176752384674818467669405132"
    "000568127145263560827785771342757789609173637178721468440901224953430146549"
    "585371050792279689258923542019956112129021960864034418159813629774771309960"
    "518707211349999998372978049951059731732816096318595024459455346908302642522"
    "308253344685035261931188171010003137838752886587533208381420617177669147303"
    "598253490428755468731159562863882353787593751957781857780532171226806613001"
    "9278766111959092164201989"
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def compute_pi_digits(num_digits: int = NUM_DIGITS) -> tuple[list[int], str]:
    try:
        import mpmath as _mp  # noqa: WPS433

        _mp.mp.dps = num_digits + 10
        pi_str = _mp.nstr(_mp.pi, num_digits + 2)
        digits_str = pi_str.split(".")[1][:num_digits]
        digits = [int(ch) for ch in digits_str]
        if len(digits) == num_digits:
            return digits, "mpmath"
    except Exception:
        pass
    base = [int(ch) for ch in _PI_1000]
    digits: list[int] = []
    while len(digits) < num_digits:
        digits.extend(base[: min(num_digits - len(digits), len(base))])
    return digits[:num_digits], "hardcoded_fallback"


def build_noise_sources(
    pi_digits: list[int], uniform_seed: int, resample_seed: int
) -> dict[str, list]:
    inv = [10 - d if d != 0 else 10 for d in pi_digits]
    nonzero = [d for d in pi_digits if d != 0]
    target_mean = float(np.mean(nonzero)) if nonzero else 5.0
    inv_mean = float(np.mean(inv))
    scale = target_mean / inv_mean if inv_mean > 0 else 1.0
    inv_matched = [d * scale for d in inv]

    rng_u = np.random.default_rng(uniform_seed)
    uniform = [int(v) for v in rng_u.integers(1, 11, size=len(pi_digits))]

    arr = np.array(pi_digits, dtype=np.int64)
    nonzero_arr = arr[arr != 0]
    zero_mask = arr == 0
    if zero_mask.any() and nonzero_arr.size > 0:
        rng_r = np.random.default_rng(resample_seed)
        arr = arr.copy()
        arr[zero_mask] = rng_r.choice(nonzero_arr, size=int(zero_mask.sum()), replace=True)
    pi_resampled = [int(v) for v in arr]

    return {
        "pi": pi_digits,
        "inverse_pi": inv,
        "inverse_pi_mean_matched": inv_matched,
        "uniform_nonzero": uniform,
        "pi_nonzero_resampled": pi_resampled,
    }


def potential(xy: np.ndarray) -> np.ndarray:
    x = xy[..., 0]
    y = xy[..., 1]
    return x * x + y * y + A_AMPLITUDE * (2.0 - np.cos(2.0 * np.pi * x) - np.cos(2.0 * np.pi * y))


def potential_grid(n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lin = np.linspace(-BOUND, BOUND, n)
    X, Y = np.meshgrid(lin, lin)
    pts = np.stack([X, Y], axis=-1)
    Z = potential(pts)
    return X, Y, Z


@dataclass
class BatchResult:
    success: np.ndarray
    arrival: np.ndarray
    final_distance: np.ndarray
    demo_path: np.ndarray


def run_batch(
    noise_digits: list,
    n_trials: int,
    steps: int,
    entropy: int,
) -> BatchResult:
    digits = np.asarray(noise_digits, dtype=float)
    n_d = len(digits)
    ss = np.random.SeedSequence(entropy=entropy)
    children = ss.spawn(n_trials + 1)
    demo_rng = np.random.default_rng(children[0])
    trial_rngs = [np.random.default_rng(c) for c in children[1:]]

    pos = np.zeros((n_trials, 2), dtype=float)
    f_val = np.zeros(n_trials, dtype=float)
    min_dist = np.full(n_trials, np.inf, dtype=float)
    arrived = np.zeros(n_trials, dtype=bool)

    for t in range(n_trials):
        rng = trial_rngs[t]
        p = rng.uniform(-BOUND, BOUND, size=2)
        pos[t] = p
        f_val[t] = float(potential(p.reshape(1, 2))[0])
        min_dist[t] = float(np.linalg.norm(p))

    T = float(T0)
    for i in range(steps):
        d_idx = i % n_d
        digit = float(digits[d_idx])
        directions = np.stack(
            [trial_rngs[t].standard_normal(2) for t in range(n_trials)], axis=0
        )
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        directions /= norms
        step = digit * STEP_SCALE * (T / T0 + 0.01)
        prop = pos + directions * step
        prop = np.clip(prop, -BOUND, BOUND)
        f_new = potential(prop)
        delta = f_new - f_val
        accept = (delta < 0) | (
            np.array([trial_rngs[t].random() for t in range(n_trials)])
            < np.exp(-np.maximum(delta, 0) / max(T, 0.001))
        )
        pos = np.where(accept[:, None], prop, pos)
        f_val = np.where(accept, f_new, f_val)
        dist = np.linalg.norm(pos, axis=1)
        min_dist = np.minimum(min_dist, dist)
        arrived |= dist < TOL
        T = max(T * COOLING, 0.001)

    final_dist = np.linalg.norm(pos, axis=1)
    success = final_dist < TOL

    # demo trajectory (1 trial)
    demo_pos = demo_rng.uniform(-BOUND, BOUND, size=2)
    path = [demo_pos.copy()]
    T = float(T0)
    for i in range(min(steps, 800)):
        digit = float(digits[i % n_d])
        direction = demo_rng.standard_normal(2)
        nrm = np.linalg.norm(direction)
        if nrm > 0:
            direction /= nrm
        step_vec = direction * digit * STEP_SCALE * (T / T0 + 0.01)
        demo_pos = np.clip(demo_pos + step_vec, -BOUND, BOUND)
        path.append(demo_pos.copy())
        T = max(T * COOLING, 0.001)
    demo_path = np.array(path)

    return BatchResult(
        success=success.astype(bool),
        arrival=arrived.astype(bool),
        final_distance=final_dist,
        demo_path=demo_path,
    )


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def two_proportion_test(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return float("nan"), 1.0
    p1, p2 = s1 / n1, s2 / n2
    p_pool = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1 - p2, 1.0
    z = (p1 - p2) / se
    p = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return p1 - p2, p


def bootstrap_diff_ci(
    a_ok: np.ndarray, b_ok: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    n_a, n_b = len(a_ok), len(b_ok)
    if n_a == 0 or n_b == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, n_a, size=n_a)
        ib = rng.integers(0, n_b, size=n_b)
        diffs.append(float(a_ok[ia].mean() - b_ok[ib].mean()))
    diffs_arr = np.array(diffs)
    return (
        float(np.mean(diffs_arr)),
        float(np.quantile(diffs_arr, 0.025)),
        float(np.quantile(diffs_arr, 0.975)),
    )


def verdict_text(p_bonf: float) -> str:
    return "significant" if p_bonf < ALPHA else "not significant"


def summarize_condition(name: str, br: BatchResult) -> dict:
    n = len(br.success)
    s = int(br.success.sum())
    a = int(br.arrival.sum())
    rate = s / n if n else 0.0
    lo, hi = wilson_interval(s, n)
    return {
        "noise": name,
        "label": NOISE_LABELS[name],
        "n_trials": n,
        "success_count": s,
        "success_rate": rate,
        "arrival_rate": a / n if n else 0.0,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "mean_final_distance": float(br.final_distance.mean()),
        "median_final_distance": float(np.median(br.final_distance)),
    }


def build_comparisons(summaries: dict[str, dict], batches: dict[str, BatchResult], n_bonf: int) -> list[dict]:
    rows = []
    for a_name, b_name, label in COMPARISON_PAIRS:
        sa, sb = summaries[a_name], summaries[b_name]
        diff, p_raw = two_proportion_test(
            sa["success_count"], sa["n_trials"], sb["success_count"], sb["n_trials"]
        )
        p_bonf = min(1.0, p_raw * n_bonf)
        rng = np.random.default_rng(SEED + hash((a_name, b_name)) % 10_000)
        b_mean, b_lo, b_hi = bootstrap_diff_ci(
            batches[a_name].success.astype(float),
            batches[b_name].success.astype(float),
            n_boot=400,
            rng=rng,
        )
        rows.append(
            {
                "comparison": label,
                "diff": diff,
                "p_raw": p_raw,
                "p_bonf": p_bonf,
                "verdict": verdict_text(p_bonf),
                "bootstrap_diff_mean": b_mean,
                "bootstrap_lo": b_lo,
                "bootstrap_hi": b_hi,
            }
        )
    return rows


def apply_dark(ax: plt.Axes) -> None:
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=DARK_TEXT)
    ax.xaxis.label.set_color(DARK_TEXT)
    ax.yaxis.label.set_color(DARK_TEXT)
    ax.title.set_color(DARK_TEXT)
    for spine in ax.spines.values():
        spine.set_color(DARK_GRID)


def plot_dashboard(
    summaries: list[dict],
    batches: dict[str, BatchResult],
    comparisons: list[dict],
    trials: int,
    steps: int,
    out_path: Path,
    title_mode: str = "single",
    seed_note: str = "",
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": DARK_BG,
            "text.color": DARK_TEXT,
            "axes.edgecolor": DARK_GRID,
            "grid.color": DARK_GRID,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if title_mode == "multiseed":
        fig.suptitle(
            "Multi-seed replication: inverse-pi noise on a rotation-symmetric potential\n"
            f"seeds: {MULTISEED} | trials/seed: {trials} | pooled trials/condition: {trials * len(MULTISEED)} | "
            f"steps: {steps} | T0={T0} | cooling={COOLING} | tol={TOL}\n"
            "POOLED = primary analysis; per-seed = replication check",
            fontsize=10,
            color=DARK_TEXT,
            y=0.98,
        )
    else:
        fig.suptitle(
            "Inverse-π noise on a rotation-symmetric potential (exploratory, not a proof)\n"
            f"A={A_AMPLITUDE:.0f}, bound=[-{BOUND:.0f},{BOUND:.0f}]², T0={T0:g}, cooling={COOLING}, "
            f"steps={steps}, trials={trials}, tol={TOL}\n"
            f"success = FINAL position within tol{seed_note}",
            fontsize=10,
            color=DARK_TEXT,
            y=0.98,
        )

    X, Y, Z = potential_grid()
    ax0 = axes[0, 0]
    pcm = ax0.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
    for name in NOISE_TYPES:
        path = batches[name].demo_path
        ax0.plot(
            path[:, 0],
            path[:, 1],
            color=NOISE_COLORS[name],
            lw=1.2,
            alpha=0.85,
            label=NOISE_LABELS[name],
        )
    circle = plt.Circle((0, 0), TOL, fill=False, color="#ff66aa", lw=1.0, linestyle="--")
    ax0.add_patch(circle)
    ax0.text(0.02, 0.02, "tol ring (true scale)", transform=ax0.transAxes, fontsize=7, color="#ff66aa")
    ax0.set_xlim(-BOUND, BOUND)
    ax0.set_ylim(-BOUND, BOUND)
    ax0.set_title("Potential f(x,y) + one demo trial per condition", fontsize=9)
    ax0.legend(fontsize=6, loc="upper right", facecolor=DARK_BG, edgecolor=DARK_GRID)
    apply_dark(ax0)

    ax1 = axes[0, 1]
    labels = [s["label"] for s in summaries]
    rates = [s["success_rate"] for s in summaries]
    xpos = np.arange(len(labels))
    bars = ax1.bar(
        xpos,
        rates,
        color=[NOISE_COLORS[t] for t in NOISE_TYPES],
        edgecolor=DARK_GRID,
    )
    for i, s in enumerate(summaries):
        err = [
            [s["success_rate"] - s["wilson_lo"]],
            [s["wilson_hi"] - s["success_rate"]],
        ]
        ax1.errorbar(i, s["success_rate"], yerr=err, fmt="none", ecolor=DARK_TEXT, capsize=3)
        ax1.text(i, rates[i] + 0.02, f"{100 * rates[i]:.0f}%", ha="center", fontsize=8, color=DARK_TEXT)
    arr_rates = [s["arrival_rate"] for s in summaries]
    ax1.plot(xpos, arr_rates, "D--", color="#ff66aa", lw=1, markersize=5, label="arrival rate (reference)")
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax1.set_ylabel("success rate")
    ax1.set_ylim(0, max(0.85, max(rates + arr_rates) * 1.25))
    ax1.set_title("Success rate (final within tol) + Wilson 95%CI", fontsize=9)
    ax1.legend(fontsize=7, facecolor=DARK_BG, edgecolor=DARK_GRID)
    ax1.grid(True, alpha=0.25)
    apply_dark(ax1)

    ax2 = axes[1, 0]
    mean_d = [s["mean_final_distance"] for s in summaries]
    ax2.bar(xpos, mean_d, color=[NOISE_COLORS[t] for t in NOISE_TYPES], edgecolor=DARK_GRID)
    for i, v in enumerate(mean_d):
        ax2.text(i, v * 1.02, f"{v:.2f}", ha="center", fontsize=8, color=DARK_TEXT)
    ax2.axhline(TOL, color=DARK_SUBTLE, ls="--", lw=1, label="tol")
    ax2.set_yscale("log")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax2.set_ylabel("mean |x_final - 0|")
    ax2.set_title("Mean final distance to the global minimum (log)", fontsize=9)
    ax2.legend(fontsize=7, facecolor=DARK_BG)
    ax2.grid(True, alpha=0.25)
    apply_dark(ax2)

    ax3 = axes[1, 1]
    ax3.axvline(0, color=DARK_SUBTLE, ls="--", lw=1)
    yoff = 0.0
    for row in comparisons:
        ax3.errorbar(
            row["diff"],
            yoff,
            xerr=[[row["diff"] - row["bootstrap_lo"]], [row["bootstrap_hi"] - row["diff"]]],
            fmt="o",
            color="#ffdd00",
            ecolor="#ffdd00",
            capsize=4,
        )
        ax3.plot(
            [row["diff"] - 1.96 * 0.05, row["diff"] + 1.96 * 0.05],
            [yoff, yoff],
            color="#00ccff",
            lw=2,
            alpha=0.5,
        )
        ax3.text(
            row["diff"] + 0.002,
            yoff,
            f"{row['comparison']}\n{row['diff']:+.4f}, p_bonf={row['p_bonf']:.3f} ({row['verdict']})",
            fontsize=7,
            color=DARK_TEXT,
            va="center",
        )
        yoff += 1.0
    ax3.set_xlim(-0.15, 0.15)
    ax3.set_yticks([])
    ax3.set_xlabel("difference in success rate")
    ax3.set_title("Success-rate difference (a - b). CI covering 0 = no clear gap", fontsize=9)
    ax3.grid(True, alpha=0.25)
    apply_dark(ax3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)


def run_experiment(
    trials: int,
    steps: int,
    seeds: list[int],
    tag: str,
) -> dict:
    pi_digits, pi_src = compute_pi_digits(NUM_DIGITS)
    all_summaries: list[dict] = []
    per_seed_rows: list[dict] = []
    pooled_batches: dict[str, BatchResult] | None = None
    pooled_summaries: dict[str, dict] = {}

    for seed in seeds:
        sources = build_noise_sources(pi_digits, uniform_seed=seed, resample_seed=seed + 1)
        batches: dict[str, BatchResult] = {}
        seed_summaries: dict[str, dict] = {}
        for j, name in enumerate(NOISE_TYPES):
            ss = np.random.SeedSequence(entropy=seed, spawn_key=(hash(name) % 2**31, j))
            br = run_batch(sources[name], trials, steps, entropy=int(ss.generate_state(1)[0]))
            batches[name] = br
            sm = summarize_condition(name, br)
            sm["seed"] = seed
            seed_summaries[name] = sm
            per_seed_rows.append(sm)

        if pooled_batches is None:
            pooled_batches = batches
            pooled_summaries = seed_summaries
        else:
            for name in NOISE_TYPES:
                pb = pooled_batches[name]
                nb = batches[name]
                pooled_batches[name] = BatchResult(
                    success=np.concatenate([pb.success, nb.success]),
                    arrival=np.concatenate([pb.arrival, nb.arrival]),
                    final_distance=np.concatenate([pb.final_distance, nb.final_distance]),
                    demo_path=pb.demo_path,
                )

    assert pooled_batches is not None
    summaries = [summarize_condition(n, pooled_batches[n]) for n in NOISE_TYPES]
    n_bonf = len(COMPARISON_PAIRS)
    comparisons = build_comparisons({s["noise"]: s for s in summaries}, pooled_batches, n_bonf)

    payload = {
        "title": "rotation_minimal — rotation-symmetric potential (toy, suggestive)",
        "tone": "示唆のみ。証明・物理法則の確定とは書かない。",
        "params": {
            "A": A_AMPLITUDE,
            "bound": BOUND,
            "T0": T0,
            "cooling": COOLING,
            "steps": steps,
            "trials_per_seed": trials,
            "seeds": seeds,
            "tol": TOL,
            "step_scale": STEP_SCALE,
            "pi_digits_source": pi_src,
        },
        "summaries": summaries,
        "per_seed": per_seed_rows,
        "comparisons": comparisons,
        "recovery_note": "rotation_minimal.py は 2026-11-25 に pyc・既存図から復旧（推定含む）",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"rotation_minimal{tag}.json"
    csv_path = RESULTS_DIR / f"rotation_minimal{tag}.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["record_type", "noise", "seed", "success_rate", "arrival_rate", "mean_final_distance"])
        for row in per_seed_rows:
            w.writerow(
                [
                    "per_seed",
                    row["noise"],
                    row.get("seed", ""),
                    f"{row['success_rate']:.6f}",
                    f"{row['arrival_rate']:.6f}",
                    f"{row['mean_final_distance']:.6f}",
                ]
            )
        for row in summaries:
            w.writerow(
                [
                    "pooled",
                    row["noise"],
                    "pooled",
                    f"{row['success_rate']:.6f}",
                    f"{row['arrival_rate']:.6f}",
                    f"{row['mean_final_distance']:.6f}",
                ]
            )

    title_mode = "multiseed" if len(seeds) > 1 else "single"
    seed_note = f" | seeds={seeds}" if len(seeds) > 1 else ""
    fig_path = FIGURES_DIR / f"rotation_minimal{tag}.png"
    plot_dashboard(
        summaries,
        pooled_batches,
        comparisons,
        trials=trials,
        steps=steps,
        out_path=fig_path,
        title_mode=title_mode,
        seed_note=seed_note,
    )
    payload["outputs"] = {"json": str(json_path.name), "csv": str(csv_path.name), "figure": str(fig_path.name)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="回転対称ポテンシャル上の逆πノイズ小規模対照（示唆）")
    parser.add_argument("--quick", action="store_true", help="短時間スモーク（20試行・500ステップ・1シード）")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--multiseed",
        action="store_true",
        help="5シード [42,7,13,29,101] で再実行（既存 5seed 図に近い構成）",
    )
    args = parser.parse_args()

    if args.quick:
        trials, steps = 20, 500
        seeds = [args.seed]
        tag = "_quick"
    elif args.multiseed:
        trials, steps = args.trials, args.steps
        seeds = list(MULTISEED)
        tag = "_multiseed"
    else:
        trials, steps = args.trials, args.steps
        seeds = [args.seed]
        tag = ""

    t0 = time.perf_counter()
    payload = run_experiment(trials, steps, seeds, tag)
    elapsed = time.perf_counter() - t0

    print("rotation_minimal.py — 回転対称ポテンシャル・逆πノイズ（玩具・示唆）")
    print(f"  trials/seed={trials}  steps={steps}  seeds={seeds}  elapsed={elapsed:.1f}s")
    for s in payload["summaries"]:
        print(
            f"  {s['label']:28s} success={s['success_rate']*100:5.1f}%  "
            f"mean|x|={s['mean_final_distance']:.3f}"
        )
    for c in payload["comparisons"]:
        print(f"  {c['comparison']}: diff={c['diff']:+.4f} p_bonf={c['p_bonf']:.4f} ({c['verdict']})")
    outs = payload.get("outputs", {})
    print(f"  -> results/{outs.get('json','')}  figures/{outs.get('figure','')}")
    print("  ※ 示唆であり証明ではありません。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())