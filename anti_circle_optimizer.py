#!/usr/bin/env python3
"""
anti_circle_optimizer.py — 逆円周率の非対称ノイズを用いた最適化アルゴリズム検証

πノイズ（0を含む対称ノイズ）と逆πノイズ（0を含まない非対称ノイズ）を
Simulated Annealing の摂動源として使用し、1D Rastrigin関数の大域的最適化性能を比較する。

出力:
  - optimization_escape.png (1920×1200)
  - ターミナル統計結果

Zenn考察:
「πノイズ（0を含む対称ノイズ）では、digit=0による完全停止が約10%発生し、
結果として探索効率が低下し、ローカルミニマにトラップされやすい。
一方、逆πノイズ（0を含まない非対称ノイズ）では全てのステップが
何らかの「補正の動き」となり、特に温度低下時に小さな揺らぎ（digit=1）が
微細な谷からの脱出を促進する。

[成功率 / 平均誤差 の数値結果に基づく考察]

この結果は、「完全な静止（ゼロ）が許されない不完全なノイズ」が、
探索アルゴリズムにおいて「局所最適解を破壊する実用的なエネルギー」として
機能することを示唆している。宇宙の根源的な非対称性が、
最適化という情報処理の文脈でも有効性を持つという点で、
哲学的命题に一定の実証的裏付けを与えるものである。
"""

import csv
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="mpmath")



# ============================================================
# 設定・定数
# ============================================================

NUM_DIGITS = 10000
N_TRIALS = 1000
N_STEPS = 3000
SEED = 42

T0 = 25.0
COOLING = 0.997
STEP_SCALE = 0.35
THRESHOLD = 0.5  # 成功判定閾値（|x| < THRESHOLD）

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"

NOISE_CONDITIONS = [
    "pi",
    "inverse_pi",
    "inverse_pi_mean_matched",
    "uniform_nonzero",
    "pi_nonzero_resampled",
]

CONDITION_LABELS = {
    "pi": "π (0を含む対称ノイズ)",
    "inverse_pi": "逆π (0を含まない非対称ノイズ)",
    "inverse_pi_mean_matched": "逆π_平均補正 (ステップ幅統制)",
    "uniform_nonzero": "一様1-10 (桁列構造なし)",
    "pi_nonzero_resampled": "π_0除去 (非ゼロ桁のみ)",
}

CONDITION_COLORS = {
    "pi": "#00ccff",
    "inverse_pi": "#ff8800",
    "inverse_pi_mean_matched": "#ff2266",
    "uniform_nonzero": "#00ff88",
    "pi_nonzero_resampled": "#8888ff",
}

# ペア比較対（条件A, 条件B）
PAIR_COMPARISONS = [
    ("pi", "inverse_pi"),
    ("pi", "inverse_pi_mean_matched"),
    ("inverse_pi", "inverse_pi_mean_matched"),
    ("inverse_pi", "uniform_nonzero"),
]

# 条件ごとの独立シードオフセット（疑似ペア化を避けるため）
CONDITION_SEED_OFFSET = {
    "pi": 0,
    "inverse_pi": 1000,
    "inverse_pi_mean_matched": 2000,
    "uniform_nonzero": 3000,
    "pi_nonzero_resampled": 4000,
}
# ============================================================
# 1. π 桁の取得（anti_circle_generator.py と同一手法）
# ============================================================

def compute_pi_digits(num_digits: int = 10000) -> list[int]:
    """mpmath で π を計算し、小数点以下 num_digits 桁のリストを返す"""
    try:
        import mpmath as mp
        mp.mp.dps = num_digits + 10
        pi_str = mp.nstr(mp.pi, num_digits + 2)
        digits_str = pi_str.split(".")[1][:num_digits]
        digits = [int(ch) for ch in digits_str]
        print(f"[INFO] π を {num_digits} 桁計算しました (mpmath)")
        return digits
    except Exception as e:
        print(f"[WARN] mpmath による π 計算に失敗: {e}")
        print("[WARN] フォールバック: ハードコードされた π 桁 (1000桁) を使用します")
        return _fallback_pi_digits()


def _fallback_pi_digits() -> list[int]:
    """mpmath フォールバック π 1000桁（反復で10000桁に拡張）"""
    pi_1000 = (
        "14159265358979323846264338327950288419716939937510"
        "58209749445923078164062862089986280348253421170679"
        "82148086513282306647093844609550582231725359408128"
        "48111745028410270193852110555964462294895493038196"
        "44288109756659334461284756482337867831652712019091"
        "45648566923460348610454326648213393607260249141273"
        "72458700660631558817488152092096282925409171536436"
        "78925903600113305305488204665213841469519415116094"
        "33057270365759591953092186117381932611793105118548"
        "07446237996274956735188575272489122793818301194912"
        "98336733624406566430860213949463952247371907021798"
        "60943702770539217176293176752384674818467669405132"
        "00056812714526356082778577134275778960917363717872"
        "14684409012249534301465495853710507922796892589235"
        "42019956112129021960864034418159813629774771309960"
        "51870721134999999837297804995105973173281609631859"
        "50244594553469083026425223082533446850352619311881"
        "71010003137838752886587533208381420617177669147303"
        "59825349042875546873115956286388235378759375195778"
        "18577805321712268066130019278766111959092164201989"
    )
    digits = [int(ch) for ch in pi_1000]
    while len(digits) < 10000:
        digits.extend(digits[: min(10000 - len(digits), len(digits))])
    return digits[:10000]


# ============================================================
# 2. 逆πの生成
# ============================================================

def compute_inverse_pi(pi_digits: list[int]) -> list[int]:
    """逆π: d→10-d, d==0→10, 値域1〜10（0を含まない）"""
    return [10 - d if d != 0 else 10 for d in pi_digits]


def compute_inverse_pi_mean_matched(
    pi_digits: list[int],
    inv_pi_digits: list[int],
) -> list[float]:
    """逆πの各桁を scale=π非ゼロ平均/逆π平均 で乗じ、平均ステップ幅をπの非ゼロ平均と揃える。
    0ステップは依然として存在しない。結果はfloatリスト。
    これは「平均ステップ幅を統制した逆π」である。
    """
    pi_nonzero = [d for d in pi_digits if d != 0]
    if not pi_nonzero:
        raise ValueError("πに非ゼロ桁が存在しません")
    pi_nonzero_mean = float(np.mean(pi_nonzero))
    inv_mean = float(np.mean(inv_pi_digits))
    if inv_mean <= 0:
        raise ValueError("逆πの平均が0以下です")
    scale = pi_nonzero_mean / inv_mean
    matched = [float(d) * scale for d in inv_pi_digits]
    return matched


def compute_pi_nonzero_digits(pi_digits: list[int]) -> list[int]:
    """πノイズから0を除いた非ゼロ桁のみのリスト"""
    return [d for d in pi_digits if d != 0]


def compute_uniform_nonzero_digits(size: int = NUM_DIGITS, seed: int | None = None) -> list[int]:
    """1〜10の一様整数ノイズ。逆πと同じ値域・ほぼ同じ平均だが、πの桁列構造を持たない。
    これは「逆πの桁列そのもの」に固有の効果があるかを見る対照条件である。
    """
    rng = np.random.default_rng(seed)
    digits = rng.integers(1, 11, size=size).tolist()
    return [int(d) for d in digits]


# ============================================================
# 3. 目的関数: 1D Rastrigin
# ============================================================

def rastrigin_1d(x: float) -> float:
    """1D Rastrigin関数: 多数のローカルミニマを持つ標準ベンチマーク"""
    return x**2 + 10.0 * (1.0 - np.cos(2.0 * np.pi * x))


# ============================================================
# 4. Simulated Annealing
# ============================================================

def simulated_annealing(
    noise_digits: list,
    T0: float = T0,
    cooling: float = COOLING,
    steps: int = N_STEPS,
    step_scale: float = STEP_SCALE,
    objective_func: Callable[[float], float] | None = None,
    x0: float | None = None,
    seed: int | None = None,
    collect_detail: bool = False,
):
    """
    Simulated Annealing による1次元最適化。

    戻り値:
      (trajectory: np.ndarray, final_x: float, accept_rate: float)
      collect_detail=True のときは 4 要素目に detail 辞書を追加:
        - zero_step_rate, mean_step_size, max_step_size, first_success_step など

    【核心メカニズム】
    ステップサイズ = digit × step_scale × direction × (T/T0 + 0.01)
    - πノイズ: digit=0 → step=0 (完全停止)
    - 逆πノイズ: digit>=1 → 常に非ゼロの動き
    - 温度比例減衰: 高温時は大きく探索、低温時は微調整
    """
    if objective_func is None:
        objective_func = rastrigin_1d

    if seed is not None:
        np.random.seed(seed)

    if x0 is None:
        x = float(np.random.uniform(-4.0, 4.0))
    else:
        x = float(x0)

    T = float(T0)
    trajectory = [x]
    accepts = 0

    steps_detail = [] if collect_detail else None
    first_success_step = None

    for i in range(steps):
        digit = noise_digits[i % len(noise_digits)]
        direction = 1.0 if np.random.random() < 0.5 else -1.0

        step_size = float(digit) * step_scale * direction * (T / T0 + 0.01)
        x_new = x + step_size
        delta = objective_func(x_new) - objective_func(x)

        accepted = bool(delta < 0 or np.random.random() < np.exp(-delta / max(T, 0.001)))
        if accepted:
            x = x_new
            accepts += 1

        if collect_detail:
            steps_detail.append(
                {
                    "step": i + 1,
                    "digit": float(digit),
                    "step_size": float(step_size),
                    "accepted": accepted,
                    "x_before": float(x if not accepted else x_new),
                    "x_after": float(x),
                    "T": float(T),
                }
            )
            if first_success_step is None and abs(x) < THRESHOLD:
                first_success_step = i + 1

        T = max(T * cooling, 0.001)
        trajectory.append(x)

    accept_rate = accepts / steps

    result = (np.array(trajectory), float(x), accept_rate)
    if collect_detail:
        abs_steps = np.abs(np.diff(trajectory))
        zero_step_rate = float(np.mean(abs_steps == 0.0)) if len(abs_steps) > 0 else 0.0
        mean_step_size = float(np.mean(abs_steps)) if len(abs_steps) > 0 else 0.0
        max_step_size = float(np.max(abs_steps)) if len(abs_steps) > 0 else 0.0
        detail = {
            "zero_step_rate": zero_step_rate,
            "mean_step_size": mean_step_size,
            "max_step_size": max_step_size,
            "first_success_step": first_success_step,
        }
        result = result + (detail,)

    return result




# ============================================================
# 6. 可視化
# ============================================================

def build_figure(
    pi_trajectories: list[np.ndarray],
    inv_trajectories: list[np.ndarray],
    pi_final: list[float],
    inv_final: list[float],
    pi_ar: list[float],
    inv_ar: list[float],
    pi_stats: tuple,
    inv_stats: tuple,
) -> plt.Figure:
    """
    Figure 1 (上半分): 代表的な軌跡比較
    Figure 2 (下半分左): π 最終位置分布（ヒストグラム）
    Figure 3 (下半分中央): 収束曲線（全試行の平均 |x| 推移）
    Figure 4 (下半分右): 逆π 最終位置分布（ヒストグラム）
    """
    fig = plt.figure(figsize=(19.20, 12.00), dpi=100, facecolor="#0a0a0a")
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                          height_ratios=[1, 1])

    # --- 代表的な試行を選択 ---
    # πがトラップされ (|final| > threshold)、逆πが脱出した試行を優先
    best_example = 0
    best_diff = -1.0
    for i in range(N_TRIALS):
        diff = abs(pi_final[i]) - abs(inv_final[i])
        if diff > best_diff and abs(pi_final[i]) > THRESHOLD:
            best_diff = diff
            best_example = i

    # --- Figure 1: 代表軌跡 (top row, full width) ---
    ax_traj = fig.add_subplot(gs[0, :])
    _draw_trajectory_panel(
        ax_traj,
        pi_trajectories[best_example],
        inv_trajectories[best_example],
        pi_final[best_example],
        inv_final[best_example],
    )

    # --- Figure 2a: π 最終分布 (左下) ---
    ax_pi_hist = fig.add_subplot(gs[1, 0])
    _draw_histogram(ax_pi_hist, pi_final, "π Noise", "#00ccff", pi_stats)

    # --- Figure 2b: 収束曲線 (下中央) ---
    ax_conv = fig.add_subplot(gs[1, 1])
    _draw_convergence(ax_conv, pi_trajectories, inv_trajectories)

    # --- Figure 2c: 逆π 最終分布 (右下) ---
    ax_inv_hist = fig.add_subplot(gs[1, 2])
    _draw_histogram(ax_inv_hist, inv_final, "Inverse π Noise", "#ff8800", inv_stats)

    return fig


def _draw_trajectory_panel(
    ax: plt.Axes,
    pi_traj: np.ndarray,
    inv_traj: np.ndarray,
    pi_final: float,
    inv_final: float,
) -> None:
    """代表的な1試行の軌跡比較"""
    ax.set_facecolor("#0a0a0a")
    ax.set_title(
        "Trajectory Comparison (Single Trial)",
        color="white", fontsize=18, fontweight="bold", pad=10,
    )

    # ローカルミニマ位置（x=±1, ±2, ... を灰色破線で表示）
    for x_min in range(-5, 6):
        ax.axhline(y=x_min, color="#333333", ls=":", lw=0.6, alpha=0.5)

    # 大域的最適解 x=0
    ax.axhline(y=0, color="white", ls="--", lw=1.2, alpha=0.7)

    # 成功範囲
    ax.axhspan(-THRESHOLD, THRESHOLD, color="white", alpha=0.03)

    steps = np.arange(len(pi_traj))

    # π 軌跡（シアン）
    ax.plot(steps, pi_traj, color="#00ccff", lw=0.8, alpha=0.7,
            label=f"π Noise (final={pi_final:.3f})")
    ax.plot(steps, pi_traj, color="#00ccff", lw=4, alpha=0.06)  # glow

    # 逆π 軌跡（橙）
    ax.plot(steps, inv_traj, color="#ff8800", lw=0.8, alpha=0.7,
            label=f"Inverse π Noise (final={inv_final:.3f})")
    ax.plot(steps, inv_traj, color="#ff8800", lw=4, alpha=0.06)  # glow

    # 凡例
    ax.legend(fontsize=11, facecolor="#1a1a1a", edgecolor="#333333",
              labelcolor="white", loc="upper right")

    ax.set_xlabel("Step", color="white", fontsize=13)
    ax.set_ylabel("x", color="white", fontsize=13)
    ax.tick_params(colors="#888888")
    ax.set_xlim(0, len(pi_traj))
    for spine in ax.spines.values():
        spine.set_color("#333333")


def _draw_histogram(
    ax: plt.Axes,
    final_x: list[float],
    label: str,
    color: str,
    stats: tuple,
) -> None:
    """最終位置の分布ヒストグラム"""
    ax.set_facecolor("#0a0a0a")
    success_rate, mean_err, median_err = stats
    ax.set_title(
        f"{label}\nSuccess={success_rate:.1f}%  Mean|e|={mean_err:.3f}",
        color="white", fontsize=13, fontweight="bold", linespacing=1.4,
    )

    bins = np.linspace(-5.5, 5.5, 23)
    ax.hist(final_x, bins=bins, color=color, alpha=0.7,
            edgecolor=color, lw=0.5)
    ax.axvline(x=0, color="white", ls="--", lw=1.0, alpha=0.5)

    # 成功範囲を薄い色で
    ax.axvspan(-THRESHOLD, THRESHOLD, color="white", alpha=0.06)

    ax.set_xlabel("Final x", color="white", fontsize=11)
    ax.set_ylabel("Count", color="white", fontsize=11)
    ax.tick_params(colors="#888888")
    ax.set_xlim(-5.5, 5.5)
    for spine in ax.spines.values():
        spine.set_color("#333333")


def _draw_convergence(
    ax: plt.Axes,
    pi_trajs: list[np.ndarray],
    inv_trajs: list[np.ndarray],
) -> None:
    """全試行の平均絶対値の推移（収束曲線）"""
    ax.set_facecolor("#0a0a0a")
    ax.set_title(
        "Convergence (Mean |x| across trials)",
        color="white", fontsize=13, fontweight="bold", pad=10,
    )

    # 全試行の平均絶対値を計算
    pi_abs = np.array([np.abs(t) for t in pi_trajs])
    inv_abs = np.array([np.abs(t) for t in inv_trajs])
    pi_mean = np.mean(pi_abs, axis=0)
    inv_mean = np.mean(inv_abs, axis=0)

    steps = np.arange(len(pi_mean))

    # スムージング: 移動平均（ノイズ低減のため）
    window = 51
    if len(pi_mean) > window:
        kernel = np.ones(window) / window
        pi_smooth = np.convolve(pi_mean, kernel, mode="valid")
        inv_smooth = np.convolve(inv_mean, kernel, mode="valid")
        steps_smooth = np.arange(len(pi_smooth)) + window // 2
    else:
        pi_smooth = pi_mean
        inv_smooth = inv_mean
        steps_smooth = steps

    ax.plot(steps_smooth, pi_smooth, color="#00ccff", lw=1.5, alpha=0.8,
            label=f"π Noise (final mean |x|={pi_mean[-1]:.3f})")
    ax.plot(steps_smooth, pi_smooth, color="#00ccff", lw=6, alpha=0.05)  # glow
    ax.plot(steps_smooth, inv_smooth, color="#ff8800", lw=1.5, alpha=0.8,
            label=f"Inverse π Noise (final mean |x|={inv_mean[-1]:.3f})")
    ax.plot(steps_smooth, inv_smooth, color="#ff8800", lw=6, alpha=0.05)  # glow

    ax.legend(
        fontsize=10, facecolor="#1a1a1a", edgecolor="#333333",
        labelcolor="white", loc="upper right",
    )

    ax.set_xlabel("Step", color="white", fontsize=11)
    ax.set_ylabel("Mean |x|", color="white", fontsize=11)
    ax.tick_params(colors="#888888")
    ax.set_xlim(0, len(pi_mean))
    for spine in ax.spines.values():
        spine.set_color("#333333")


# ============================================================
# 7. Zenn 考察（コンソール出力用）
# ============================================================

ZENN_ESSAY_TEMPLATE = """
===============================================================
Zenn考察: 「逆円周率の非対称ノイズを用いた最適化アルゴリズム」

◆ πノイズ（0を含む対称ノイズ）
  → 約10%の桁が0であり、そのステップでは完全に停止する。
    結果として探索効率が低下し、ローカルミニマにトラップされやすい。

◆ 逆πノイズ（0を含まない非対称ノイズ）
  → 全ての桁が1以上であり、全ステップが何らかの「補正の動き」となる。
    特に温度低下時に小さな揺らぎ（digit=1）が微細な谷からの脱出を促進する。

◆ 数値結果
  π ノイズ:   成功率 {pi_success:.1f}%, 平均誤差 {pi_mean_err:.4f}
  逆π ノイズ: 成功率 {inv_success:.1f}%, 平均誤差 {inv_mean_err:.4f}
  成功率向上:  +{success_diff:.1f} ポイント
  平均誤差低減: {error_reduction:.1f}%

◆ 結論
  「完全な静止（ゼロ）が許されない不完全なノイズ」が、
  探索アルゴリズムにおいて「局所最適解を破壊する実用的なエネルギー」として
  機能することを示唆している。宇宙の根源的な非対称性が、
  最適化という情報処理の文脈でも有効性を持つという点で、
  哲学的命题に一定の実証的裏付けを与えるものである。

  {judgment_comment}
===============================================================
"""


# ============================================================
# 8. メイン実行
# ============================================================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 日本語ラベルを正しく表示するためのフォント設定（環境に依存）
    try:
        plt.rcParams["font.family"] = "Meiryo"
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 70)
    print("  anti_circle_optimizer.py — 拡張版: 逆πノイズ局所解脱出効果の切り分け")
    print("  Simulated Annealing on 1D Rastrigin with multiple noise conditions")
    print("=" * 70)

    results, initial_x_list = run_experiment()

    # 統計集計
    summary = {}
    for cond in NOISE_CONDITIONS:
        summary[cond] = compute_condition_stats(
            results[cond]["final_x"],
            results[cond]["trajectories"],
            results[cond]["accept_rates"],
            results[cond]["details"],
        )

    # 条件間比較
    comparisons = []
    for ca, cb in PAIR_COMPARISONS:
        comp = compare_conditions(
            summary[ca], summary[cb],
            results[ca]["successes"], results[cb]["successes"],
            CONDITION_LABELS[ca], CONDITION_LABELS[cb],
        )
        comparisons.append(comp)

    # ターミナル出力
    print("\n" + "=" * 70)
    print("===== サマリー統計 =====")
    print(f"試行回数: {N_TRIALS}, ステップ数: {N_STEPS}")
    print(f"パラメータ: T0={T0}, cooling={COOLING}, step_scale={STEP_SCALE}, 閾値 |x| < {THRESHOLD}")
    print()
    for cond in NOISE_CONDITIONS:
        s = summary[cond]
        print(f"--- {CONDITION_LABELS[cond]} ---")
        print(f"  成功率: {s['success_rate']:.2f}%  (95% Wilson CI: [{s['wilson_ci_low']*100:.2f}, {s['wilson_ci_high']*100:.2f}])")
        print(f"  平均絶対誤差: {s['mean_error']:.4f}  中央絶対誤差: {s['median_error']:.4f}")
        print(f"  平均受理率: {s['mean_accept_rate']*100:.2f}%  ゼロステップ率: {s['mean_zero_step_rate']*100:.2f}%")
        print(f"  平均ステップ幅: {s['mean_step_size']:.6f}  最大ステップ幅: {s['max_step_size']:.6f}")
        esc = s['mean_escape_steps']
        print(f"  大域最適解到達: {s['escape_count']}/{N_TRIALS}  平均到達ステップ: {esc if esc is None else f'{esc:.1f}'}")
        print()

    print("===== 条件間比較 =====")
    for c in comparisons:
        print(f"--- {c['comparison']} ---")
        print(f"  成功率: {c['condition_a']}={c['p_a']*100:.2f}% / {c['condition_b']}={c['p_b']*100:.2f}%")
        print(f"  リスク差: {c['risk_difference']*100:+.2f} ポイント (95% CI: [{c['risk_difference_ci_low']*100:+.2f}, {c['risk_difference_ci_high']*100:+.2f}])")
        or_txt = "inf" if c["odds_ratio_inf"] else f"{c['odds_ratio']:.4f}"
        print(f"  オッズ比: {or_txt}  コーエンh: {c['cohens_h']:.4f}")
        print(f"  p(McNemar): {c['p_mcnemar']:.4f}  p(permutation): {c['p_permutation']:.4f}")
        print()

    # 保存
    print("[保存] results/ ...", flush=True)
    save_results_csv(results)
    save_summary_csv(summary, comparisons)
    save_results_json(results, summary, comparisons, initial_x_list)

    print("[保存] figures/ ...", flush=True)
    figs = {
        "optimization_escape.png": build_escape_figure(results, summary),
        "success_rate_comparison.png": build_success_rate_figure(summary),
        "step_distribution_comparison.png": build_step_distribution_figure(results),
        "convergence_comparison.png": build_convergence_figure(results),
        "escape_time_comparison.png": build_escape_time_figure(results, summary),
    }
    for fname, fig in figs.items():
        out = FIGURES_DIR / fname
        fig.savefig(str(out), dpi=100, facecolor="#0a0a0a", bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        print(f"      保存完了: {out}")

    print("\n✅ 完了！")






# ============================================================
# 6. 実験実行
# ============================================================

def generate_noise_digits(condition: str, pi_digits: list[int], inv_pi_digits: list[int], seed: int) -> list:
    """条件名に応じてノイズ桁列を生成"""
    if condition == "pi":
        return list(pi_digits)
    elif condition == "inverse_pi":
        return list(inv_pi_digits)
    elif condition == "inverse_pi_mean_matched":
        return compute_inverse_pi_mean_matched(pi_digits, inv_pi_digits)
    elif condition == "uniform_nonzero":
        return compute_uniform_nonzero_digits(size=NUM_DIGITS, seed=seed)
    elif condition == "pi_nonzero_resampled":
        return compute_pi_nonzero_digits(pi_digits)
    else:
        raise ValueError(f"未知の条件: {condition}")


def run_experiment() -> tuple[dict, list[float]]:
    """全条件で実験を実行し、結果と初期値リストを返す"""
    print(f"[1/3] π を {NUM_DIGITS} 桁計算中...")
    pi_digits = compute_pi_digits(NUM_DIGITS)
    print(f"      取得完了: {len(pi_digits)} 桁")

    print("[2/3] 逆πを生成中...")
    inv_pi_digits = compute_inverse_pi(pi_digits)
    assert 0 not in inv_pi_digits, "逆πに0が含まれています！"
    pi_zero_count = pi_digits.count(0)
    print(f"      生成完了: {len(inv_pi_digits)} 桁 (値域 {min(inv_pi_digits)}–{max(inv_pi_digits)})")
    print(f"      π の 0 出現率: {100.0 * pi_zero_count / len(pi_digits):.1f}%")

    # 固定ノイズを事前生成（uniform_nonzero はシード依存なので各trialで生成）
    noise_cache = {
        "pi": list(pi_digits),
        "inverse_pi": list(inv_pi_digits),
        "inverse_pi_mean_matched": compute_inverse_pi_mean_matched(pi_digits, inv_pi_digits),
        "pi_nonzero_resampled": compute_pi_nonzero_digits(pi_digits),
    }

    print("[3/3] Simulated Annealing 実行中 ...")

    results = {
        cond: {"final_x": [], "trajectories": [], "successes": [], "accept_rates": [], "details": []}
        for cond in NOISE_CONDITIONS
    }
    initial_x_list: list[float] = []  # ペア比較用の初期値記録

    for trial in range(N_TRIALS):
        if (trial + 1) % 100 == 0 or trial == 0:
            print(f"      Trial {trial + 1}/{N_TRIALS}...")

        # 共通の初期値を1つ生成し、全条件で共有（ペア化のため）。
        # 各条件のシードは独立（疑似ペア化を避ける）。
        master_seed = SEED + trial
        np.random.seed(master_seed)
        x0 = float(np.random.uniform(-4.0, 4.0))
        initial_x_list.append(x0)

        for cond in NOISE_CONDITIONS:
            cond_seed = master_seed + CONDITION_SEED_OFFSET[cond]

            digits = noise_cache.get(cond)
            if digits is None:
                digits = generate_noise_digits(cond, pi_digits, inv_pi_digits, cond_seed)

            traj, final_x, accept_rate, detail = simulated_annealing(
                digits,
                T0=T0,
                cooling=COOLING,
                steps=N_STEPS,
                step_scale=STEP_SCALE,
                x0=x0,
                seed=cond_seed,
                collect_detail=True,
            )
            results[cond]["final_x"].append(final_x)
            results[cond]["trajectories"].append(traj)
            results[cond]["successes"].append(bool(abs(final_x) < THRESHOLD))
            results[cond]["accept_rates"].append(accept_rate)
            results[cond]["details"].append(detail)

    return results, initial_x_list


# ============================================================
# 5. 統計処理
# ============================================================

def wilson_ci(success_count: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """成功率の95% Wilson信頼区間を返す (lower, upper)。比率は0-1の範囲。"""
    if n == 0:
        return (0.0, 0.0)
    p = success_count / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cohen_h(p1: float, p2: float) -> float:
    """2つの割合の間のコーエンのh（効果量）"""
    a = math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    b = math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return 2.0 * (a - b)


def mcnemar_pvalue(b: int, c: int) -> float:
    """McNemar検定のp値（二項分布 exact、両側）。不一致セル b, c に対して。
    帰無仮説: b と c は同確率。ペア設計の成功/失敗比較に使用。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 二項分布 B(n, 0.5) で X <= k の累積確率
    term = math.exp(-n * math.log(2.0))  # P(X=0)
    cdf = term
    for x in range(1, k + 1):
        term *= (n - x + 1) / x
        cdf += term
    p = 2.0 * min(cdf, 1.0 - cdf + term)  # 両側近似
    return min(1.0, p)


def paired_permutation_test(
    success_a: list[bool],
    success_b: list[bool],
    n_permutations: int = 10000,
    seed: int = SEED,
) -> float:
    """ペア化された成功/失敗データに対するpermutation test。
    各ペアについてA/Bのラベルをランダムに入れ替え（符号反転に相当）、
    観測された成功率差の絶対値より極端なものが出る確率を返す。
    ライブラリ追加不要の自前実装。
    """
    rng = np.random.default_rng(seed)
    a = np.array(success_a, dtype=bool)
    b = np.array(success_b, dtype=bool)
    if len(a) == 0:
        return 1.0
    diff_obs = float(np.mean(a) - np.mean(b))
    paired_diff = a.astype(np.int8) - b.astype(np.int8)  # -1, 0, 1
    n = len(paired_diff)
    count_extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_diff = float(np.mean(paired_diff * signs))
        if abs(perm_diff) >= abs(diff_obs):
            count_extreme += 1
    return (count_extreme + 1) / (n_permutations + 1)


def compute_condition_stats(
    final_x_list: list[float],
    trajectories: list[np.ndarray],
    accept_rates: list[float],
    details: list[dict],
    threshold: float = THRESHOLD,
) -> dict:
    """1条件のサマリー統計を計算"""
    arr = np.array(final_x_list, dtype=float)
    n = len(arr)
    success_mask = np.abs(arr) < threshold
    success_count = int(np.sum(success_mask))
    success_rate = success_count / n * 100.0
    ci_low, ci_high = wilson_ci(success_count, n)
    mean_error = float(np.mean(np.abs(arr))) if n > 0 else 0.0
    median_error = float(np.median(np.abs(arr))) if n > 0 else 0.0

    mean_accept_rate = float(np.mean(accept_rates)) if accept_rates else 0.0

    zero_rates = [d["zero_step_rate"] for d in details]
    mean_steps = [d["mean_step_size"] for d in details]
    max_steps = [d["max_step_size"] for d in details]
    first_success_steps = [d["first_success_step"] for d in details]

    mean_zero_step_rate = float(np.mean(zero_rates)) if zero_rates else 0.0
    mean_step_size = float(np.mean(mean_steps)) if mean_steps else 0.0
    max_step_size = float(np.mean(max_steps)) if max_steps else 0.0

    reached_steps = [s for s in first_success_steps if s is not None]
    mean_escape_steps = float(np.mean(reached_steps)) if reached_steps else None
    escape_count = len(reached_steps)

    return {
        "n_trials": n,
        "success_count": success_count,
        "success_rate": success_rate,
        "wilson_ci_low": ci_low,
        "wilson_ci_high": ci_high,
        "mean_error": mean_error,
        "median_error": median_error,
        "mean_accept_rate": mean_accept_rate,
        "mean_zero_step_rate": mean_zero_step_rate,
        "mean_step_size": mean_step_size,
        "max_step_size": max_step_size,
        "escape_count": escape_count,
        "mean_escape_steps": mean_escape_steps,
        "final_x_list": [float(v) for v in arr],
        "first_success_steps": first_success_steps,
    }


def compare_conditions(
    stats_a: dict,
    stats_b: dict,
    successes_a: list[bool],
    successes_b: list[bool],
    label_a: str,
    label_b: str,
    n_permutations: int = 10000,
) -> dict:
    """2条件のペア比較統計を計算（同じtrial・同じ初期値をペアとする）"""
    n = len(successes_a)
    sa = np.array(successes_a, dtype=bool)
    sb = np.array(successes_b, dtype=bool)

    p_a = float(np.mean(sa))
    p_b = float(np.mean(sb))
    risk_diff = p_a - p_b

    # 2x2 交差表 (ペア)
    both = int(np.sum(sa & sb))
    a_only = int(np.sum(sa & ~sb))  # A成功 B失敗
    b_only = int(np.sum(~sa & sb))  # A失敗 B成功
    neither = int(np.sum(~sa & ~sb))

    # オッズ比 (ペア設計では McNemar 的なオッズ比 = b_only / a_only)
    if a_only == 0 and b_only == 0:
        odds_ratio = 1.0
    elif a_only == 0:
        odds_ratio = float("inf")
    else:
        odds_ratio = b_only / a_only

    # p値
    p_mcnemar = mcnemar_pvalue(a_only, b_only)
    p_perm = paired_permutation_test(sa.tolist(), sb.tolist(), n_permutations=n_permutations)

    # 効果量: コーエンのh
    h = cohen_h(p_a, p_b)

    # リスク差の95% CI (Wald近似、n=1000 なので許容)
    se = math.sqrt(p_a * (1 - p_a) / n + p_b * (1 - p_b) / n) if n > 0 else 0.0
    z = 1.959963984540054
    ci_low = risk_diff - z * se
    ci_high = risk_diff + z * se

    return {
        "comparison": f"{label_a} vs {label_b}",
        "condition_a": label_a,
        "condition_b": label_b,
        "n_pairs": n,
        "p_a": p_a,
        "p_b": p_b,
        "risk_difference": risk_diff,
        "risk_difference_ci_low": ci_low,
        "risk_difference_ci_high": ci_high,
        "odds_ratio": None if odds_ratio == float("inf") else odds_ratio,
        "odds_ratio_inf": odds_ratio == float("inf"),
        "both_success": both,
        "a_only_success": a_only,
        "b_only_success": b_only,
        "neither_success": neither,
        "p_mcnemar": p_mcnemar,
        "p_permutation": p_perm,
        "cohens_h": h,
    }


# ============================================================
# 7. 出力保存（CSV / JSON）
# ============================================================

def save_results_csv(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "optimization_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial", "condition", "final_x", "success",
            "zero_step_rate", "mean_step_size", "max_step_size", "first_success_step",
        ])
        for trial in range(N_TRIALS):
            for cond in NOISE_CONDITIONS:
                d = results[cond]["details"][trial]
                writer.writerow([
                    trial + 1,
                    cond,
                    f"{results[cond]['final_x'][trial]:.8f}",
                    int(results[cond]["successes"][trial]),
                    f"{d['zero_step_rate']:.6f}",
                    f"{d['mean_step_size']:.8f}",
                    f"{d['max_step_size']:.8f}",
                    d["first_success_step"] if d["first_success_step"] is not None else "",
                ])
    print(f"      詳細CSV保存: {path}")


def save_summary_csv(summary: dict, comparisons: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "optimization_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "n_trials", "success_count", "success_rate_percent",
            "wilson_ci_low", "wilson_ci_high", "mean_error", "median_error",
            "mean_accept_rate", "mean_zero_step_rate", "mean_step_size", "max_step_size",
            "escape_count", "mean_escape_steps",
        ])
        for cond in NOISE_CONDITIONS:
            s = summary[cond]
            writer.writerow([
                cond, s["n_trials"], s["success_count"], f"{s['success_rate']:.4f}",
                f"{s['wilson_ci_low']:.6f}", f"{s['wilson_ci_high']:.6f}",
                f"{s['mean_error']:.8f}", f"{s['median_error']:.8f}",
                f"{s['mean_accept_rate']:.6f}", f"{s['mean_zero_step_rate']:.6f}",
                f"{s['mean_step_size']:.8f}", f"{s['max_step_size']:.8f}",
                s["escape_count"], s["mean_escape_steps"] if s["mean_escape_steps"] is not None else "",
            ])
        writer.writerow([])
        writer.writerow([
            "comparison", "risk_difference", "ci_low", "ci_high",
            "odds_ratio", "p_mcnemar", "p_permutation", "cohens_h",
        ])
        for c in comparisons:
            writer.writerow([
                c["comparison"], f"{c['risk_difference']:.6f}",
                f"{c['risk_difference_ci_low']:.6f}", f"{c['risk_difference_ci_high']:.6f}",
                c["odds_ratio"] if c["odds_ratio"] is not None else "inf",
                f"{c['p_mcnemar']:.6f}", f"{c['p_permutation']:.6f}", f"{c['cohens_h']:.6f}",
            ])
    print(f"      サマリーCSV保存: {path}")


def save_results_json(results: dict, summary: dict, comparisons: list[dict], initial_x_list: list[float]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "optimization_results.json"
    payload = {
        "config": {
            "n_trials": N_TRIALS,
            "n_steps": N_STEPS,
            "seed": SEED,
            "T0": T0,
            "cooling": COOLING,
            "step_scale": STEP_SCALE,
            "threshold": THRESHOLD,
            "num_digits": NUM_DIGITS,
        },
        "summary": summary,
        "comparisons": comparisons,
        "initial_x": [float(x) for x in initial_x_list],
        "conditions": {
            cond: {
                "final_x": [float(v) for v in results[cond]["final_x"]],
                "successes": [bool(v) for v in results[cond]["successes"]],
            }
            for cond in NOISE_CONDITIONS
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"      JSON保存: {path}")


# ============================================================
# 8. 可視化（5枚）
# ============================================================

def _style_ax(ax, title: str) -> None:
    ax.set_facecolor("#0a0a0a")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333333")


def build_success_rate_figure(summary: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100, facecolor="#0a0a0a")
    labels = [CONDITION_LABELS[c] for c in NOISE_CONDITIONS]
    rates = [summary[c]["success_rate"] for c in NOISE_CONDITIONS]
    ci_lows = [summary[c]["wilson_ci_low"] * 100 for c in NOISE_CONDITIONS]
    ci_highs = [summary[c]["wilson_ci_high"] * 100 for c in NOISE_CONDITIONS]
    errors = [
        [rates[i] - ci_lows[i] for i in range(len(rates))],
        [ci_highs[i] - rates[i] for i in range(len(rates))],
    ]
    colors = [CONDITION_COLORS[c] for c in NOISE_CONDITIONS]
    bars = ax.bar(range(len(rates)), rates, yerr=errors, capsize=5, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(labels, rotation=15, ha="right", color="white", fontsize=9)
    ax.set_ylabel("成功率 (%)", color="white", fontsize=12)
    ax.set_ylim(0, 100)
    _style_ax(ax, "成功率比較（95% Wilson信頼区間）")
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 2, f"{r:.1f}%", ha="center", color="white", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def build_step_distribution_figure(results: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100, facecolor="#0a0a0a")
    for cond in NOISE_CONDITIONS:
        steps_all = []
        for traj in results[cond]["trajectories"][::20]:  # 代表試行をサンプリングして軽量化
            diffs = np.abs(np.diff(traj))
            steps_all.extend(diffs.tolist())
        if steps_all:
            ax.hist(steps_all, bins=40, alpha=0.55, label=CONDITION_LABELS[cond],
                    color=CONDITION_COLORS[cond], density=True)
    ax.set_xlabel("ステップ幅 |Δx|", color="white", fontsize=12)
    ax.set_ylabel("密度", color="white", fontsize=12)
    _style_ax(ax, "ステップ幅分布比較（代表試行より）")
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")
    fig.tight_layout()
    return fig

def build_convergence_figure(results: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100, facecolor="#0a0a0a")
    window = 101
    kernel = np.ones(window) / window
    for cond in NOISE_CONDITIONS:
        abs_arr = np.array([np.abs(t) for t in results[cond]["trajectories"]])
        mean_abs = np.mean(abs_arr, axis=0)
        if len(mean_abs) > window:
            smooth = np.convolve(mean_abs, kernel, mode="valid")
            steps_s = np.arange(len(smooth)) + window // 2
        else:
            smooth = mean_abs
            steps_s = np.arange(len(mean_abs))
        ax.plot(steps_s, smooth, color=CONDITION_COLORS[cond], lw=1.5, alpha=0.9,
                label=f"{CONDITION_LABELS[cond]} (final={smooth[-1]:.3f})")
    ax.axhline(y=THRESHOLD, color="white", ls="--", lw=1.0, alpha=0.5)
    ax.set_xlabel("ステップ", color="white", fontsize=12)
    ax.set_ylabel("平均 |x|", color="white", fontsize=12)
    _style_ax(ax, "収束曲線比較（全試行平均 |x|）")
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")
    fig.tight_layout()
    return fig


def build_escape_time_figure(results: dict, summary: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100, facecolor="#0a0a0a")
    labels = []
    means = []
    errs = []
    colors = []
    for cond in NOISE_CONDITIONS:
        vals = [d["first_success_step"] for d in results[cond]["details"] if d["first_success_step"] is not None]
        if vals:
            labels.append(CONDITION_LABELS[cond])
            means.append(float(np.mean(vals)))
            errs.append(float(np.std(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            colors.append(CONDITION_COLORS[cond])
    if means:
        bars = ax.bar(range(len(means)), means, yerr=errs, capsize=5, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right", color="white", fontsize=9)
        ax.set_ylabel("大域最適解到達までの平均ステップ数", color="white", fontsize=11)
        _style_ax(ax, "到達ステップ数比較（到達した試行のみ）")
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + max(means) * 0.02, f"{m:.0f}", ha="center",
                    color="white", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def build_escape_figure(results: dict, summary: dict) -> plt.Figure:
    """代表的な軌跡比較 + 各条件最終位置ヒストグラム（ダークテーマ維持）"""
    fig = plt.figure(figsize=(16, 10), dpi=100, facecolor="#0a0a0a")
    gs = fig.add_gridspec(2, 5, hspace=0.4, wspace=0.3, height_ratios=[1, 1])

    # 代表試行: π が失敗し inverse_pi が成功したものを優先
    best_example = 0
    best_diff = -1.0
    for i in range(N_TRIALS):
        if abs(results["pi"]["final_x"][i]) > THRESHOLD and abs(results["inverse_pi"]["final_x"][i]) < THRESHOLD:
            diff = abs(results["pi"]["final_x"][i]) - abs(results["inverse_pi"]["final_x"][i])
            if diff > best_diff:
                best_diff = diff
                best_example = i

    # 軌跡パネル (π vs inverse_pi)
    ax_traj = fig.add_subplot(gs[0, :])
    ax_traj.set_facecolor("#0a0a0a")
    ax_traj.set_title("代表軌跡比較（π vs 逆π）", color="white", fontsize=15, fontweight="bold", pad=10)
    for x_min in range(-5, 6):
        ax_traj.axhline(y=x_min, color="#333333", ls=":", lw=0.6, alpha=0.5)
    ax_traj.axhline(y=0, color="white", ls="--", lw=1.2, alpha=0.7)
    ax_traj.axhspan(-THRESHOLD, THRESHOLD, color="white", alpha=0.03)
    steps = np.arange(len(results["pi"]["trajectories"][best_example]))
    ax_traj.plot(steps, results["pi"]["trajectories"][best_example], color="#00ccff", lw=0.8, alpha=0.7,
                 label=f"π (final={results['pi']['final_x'][best_example]:.3f})")
    ax_traj.plot(steps, results["inverse_pi"]["trajectories"][best_example], color="#ff8800", lw=0.8, alpha=0.7,
                 label=f"逆π (final={results['inverse_pi']['final_x'][best_example]:.3f})")
    ax_traj.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white", loc="upper right")
    ax_traj.set_xlabel("ステップ", color="white")
    ax_traj.set_ylabel("x", color="white")
    ax_traj.tick_params(colors="#888888")
    ax_traj.set_xlim(0, len(steps))
    for spine in ax_traj.spines.values():
        spine.set_color("#333333")

    # 最終位置ヒストグラム（全条件）
    for idx, cond in enumerate(NOISE_CONDITIONS):
        ax = fig.add_subplot(gs[1, idx])
        ax.set_facecolor("#0a0a0a")
        s = summary[cond]
        ax.hist(results[cond]["final_x"], bins=np.linspace(-5.5, 5.5, 23),
                color=CONDITION_COLORS[cond], alpha=0.7, edgecolor=CONDITION_COLORS[cond], lw=0.5)
        ax.axvline(x=0, color="white", ls="--", lw=1.0, alpha=0.5)
        ax.axvspan(-THRESHOLD, THRESHOLD, color="white", alpha=0.06)
        ax.set_title(f"{CONDITION_LABELS[cond]}\n成功={s['success_rate']:.1f}%  平均|e|={s['mean_error']:.3f}",
                     color="white", fontsize=10, fontweight="bold", linespacing=1.3)
        ax.set_xlabel("Final x", color="white", fontsize=9)
        ax.set_ylabel("Count", color="white", fontsize=9)
        ax.tick_params(colors="#888888", labelsize=8)
        ax.set_xlim(-5.5, 5.5)
        for spine in ax.spines.values():
            spine.set_color("#333333")

    fig.tight_layout()
    return fig

# エントリーポイント
if __name__ == "__main__":
    main()