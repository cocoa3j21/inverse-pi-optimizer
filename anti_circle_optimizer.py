#!/usr/bin/env python3
"""
anti_circle_optimizer.py — 逆πノイズ「平均ステップ幅仮説」マトリクス検証

5種類のノイズ源を Simulated Annealing (SA) の摂動源として、
  5ノイズ条件 × 6目的関数 × 4温度帯 (T0) × 3冷却率
のマトリクスを実行し、「逆πの効果」が次のどれに依存するのかを切り分ける。

  (a) 平均ステップ幅の増大  → inverse_pi_mean_matched で統制する
  (b) 温度帯 (T0)           → 5 / 15 / 25 / 40 で比較する
  (c) 目的関数の形状        → Rastrigin 1D/2D, Ackley, Griewank,
                              Rosenbrock, Sphere で比較する

比較する5条件:
  1. pi                     : π 桁そのもの（0〜9, 約10%が0=完全停止）
  2. inverse_pi             : 逆π（10-d, 1〜10, 0なし, 平均ステップ幅は大きめ）
  3. inverse_pi_mean_matched: 逆πを線形スケールし、平均ステップ幅を
                              π 非ゼロ桁の平均に一致させた統制条件
  4. uniform_nonzero        : 一様整数ノイズ 1〜10（0なし対照）
  5. pi_nonzero_resampled   : π の 0 を非ゼロ桁から復元抽出で置換（0なし・分布保持）

乱数の設計（再現可能性と独立性の両立）:
  - 実験全体の entropy は SEED 固定で再現可能。
  - 各セル (目的関数, T0, 冷却率, ノイズ条件) ごとに
    SeedSequence(entropy=SEED, spawn_key=(...)) で**独立したストリーム**を導出する。
  - 旧版のように「同一シードを条件をまたいで再設定」しない。同一シードの再利用は
    試行を artificial にペア化し、対応あり検定の前提を偽るためだ。
    よって本実験の比較はすべて**非対応 (unpaired)** 検定で行う。

出力:
  - results/optimization_results.csv     : 試行ごとの生データ
  - results/optimization_summary.csv     : セル（条件）ごとの統計サマリ
  - results/optimization_results.json    : メタ + サマリ + 比較検定 + 解釈
  - results/comparisons.csv              : ペアワイズ比較（Bonferroni 補正付き）
  - results/final_position_histogram.csv : 最終位置ヒストグラム
  - figures/optimization_escape.png
  - figures/success_rate_comparison.png
  - figures/step_distribution_comparison.png
  - figures/convergence_comparison.png
  - figures/escape_time_comparison.png

Zenn 用の書き分け（断定しない）:
  このスクリプトが返るのは p 値と信頼区間であり、真理ではない。想定される着地は
    * 平均ステップ幅を統制すると差が縮む／消える
      → 逆πの見かけの優位は「歩幅の大きさ」で説明できる可能性が示唆される。
    * 統制後も特定の温度帯・特定の関数で差が残る
      → 「0でないこと」自体の効果は、条件付きでは無害とは言い切れない。
  どちらでも「宇宙の非対称性が AI を救う」種の断定は行わない。
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
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境でも安全に描画
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="mpmath")

# Windows 標準コンソール (cp932) でも日本語出力で落ちないよう UTF-8 に寄せる。
# 失敗しても errors="replace" で継続し、実験自体を止めない。
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - 稀な環境
            pass
del _stream_name, _stream


# mpmath は任意依存（無い場合はハードコード済み桁でフォールバック）
try:
    import mpmath as _mp
    _HAS_MPMATH = True
except Exception:  # pragma: no cover - 環境依存
    _mp = None
    _HAS_MPMATH = False


# ============================================================
# 0. 実験定数
# ============================================================

N_TRIALS = 100          # 後方互換: 旧メインループの試行回数
N_STEPS = 3000          # 1試行あたりの SA ステップ数
SEED = 42               # 全実験の entropy 基準（再現可能）

T0 = 25.0               # 既定の初期温度（旧メイン互換）
COOLING = 0.997         # 既定の冷却率（旧メイン互換）
STEP_SCALE = 0.35       # ステップ幅スケール
THRESHOLD = 0.5         # 成功判定閾値（大域最適解からのユークリッド距離）

NUM_DIGITS = 10000      # 使用する π 桁数

# --- マトリクス定義 ---
MATRIX_T0_LIST: tuple[float, ...] = (5.0, 15.0, 25.0, 40.0)
MATRIX_COOLING_LIST: tuple[float, ...] = (0.995, 0.997, 0.999)
MATRIX_TRIALS = 60      # マトリクスモードの 1セルあたり試行数
CURVE_STRIDE = 25       # 収束曲線を記録する間隔
N_PERM = 10000          # 置換検定の反復回数
Z_95 = 1.959963984540054

BASE_T0 = 25.0          # 主表示（図・コンソール）に使う基準温度帯
BASE_COOLING = 0.997    # 主表示に使う基準冷却率

# 事前に登録した 4 ペアだけを比較する（後から増やして多重度を上げない）
COMPARISON_PAIRS: list[tuple[str, str]] = [
    ("pi", "inverse_pi"),
    ("pi", "inverse_pi_mean_matched"),
    ("inverse_pi", "inverse_pi_mean_matched"),
    ("inverse_pi", "uniform_nonzero"),
]

# 5つのノイズ条件（順序は可視化・出力・spawn_key の並び順として使用）
NOISE_TYPES: list[str] = [
    "pi",
    "inverse_pi",
    "inverse_pi_mean_matched",
    "uniform_nonzero",
    "pi_nonzero_resampled",
]

NOISE_LABELS: dict[str, str] = {
    "pi": "π (0-9)",
    "inverse_pi": "Inverse π (1-10)",
    "inverse_pi_mean_matched": "Inverse π (mean-matched)",
    "uniform_nonzero": "Uniform 1-10",
    "pi_nonzero_resampled": "π non-zero resampled",
}

NOISE_COLORS: dict[str, str] = {
    "pi": "#00ccff",
    "inverse_pi": "#ff8800",
    "inverse_pi_mean_matched": "#ffdd00",
    "uniform_nonzero": "#66ff99",
    "pi_nonzero_resampled": "#cc66ff",
}

NOISE_MEANINGS: dict[str, str] = {
    "pi": "対称ノイズ。約10%のステップで digit=0 → 完全停止が起こる。",
    "inverse_pi": "非対称ノイズ。0を含まず、平均ステップ幅はπより大きい。",
    "inverse_pi_mean_matched": "逆πの歩幅をπ非ゼロ平均に揃えた統制条件（本実験の主役）。",
    "uniform_nonzero": "0を含まない乱数による対照。逆π特有の分布効果を見る。",
    "pi_nonzero_resampled": "πの0だけを非ゼロ桁で埋めた対照。分布を保ちつつ0だけ消す。",
}

# --- 任意拡張の第2統制条件（既定の 5×6×4×3 マトリクスには含めない） ---
# inverse_pi_mean_matched は『動いた時の歩幅』を揃えるだけで、π側の 10% 停止を
# 差し引いた総平均歩幅までは揃えない。総移動量まで揃えたのが width_matched。
EXTRA_NOISE_TYPES: list[str] = ["inverse_pi_width_matched"]
NOISE_LABELS["inverse_pi_width_matched"] = "Inverse π (width-matched)"
NOISE_COLORS["inverse_pi_width_matched"] = "#ff66cc"
NOISE_MEANINGS["inverse_pi_width_matched"] = (
    "逆πの総平均歩幅をπ（0を含む平均）に揃えた追加統制（--with-width-matched）。"
)

# 事前登録した比較に、追加条件が存在するときだけ評価されるペアを足す
EXTRA_COMPARISON_PAIRS: list[tuple[str, str]] = [
    ("pi", "inverse_pi_width_matched"),
    ("inverse_pi_mean_matched", "inverse_pi_width_matched"),
]


def active_noise_conditions(include_extra: bool = False) -> list[str]:
    """マトリクスに実際に流し込むノイズ条件の一覧を返す。"""
    return list(NOISE_TYPES) + (list(EXTRA_NOISE_TYPES) if include_extra else [])

# 出力先
MODULE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODULE_DIR / "results"
FIGURES_DIR = MODULE_DIR / "figures"

DARK_BG = "#0a0a0a"
DARK_GRID = "#2b2b2b"
DARK_TEXT = "#e6e6e6"
DARK_SUBTLE = "#8b8b8b"


# ============================================================
# 1. π 桁の取得
# ============================================================

def compute_pi_digits(num_digits: int = NUM_DIGITS) -> list[int]:
    """mpmath で π を計算し、小数点以下 num_digits 桁のリストを返す。"""
    if _HAS_MPMATH:
        try:
            _mp.mp.dps = num_digits + 10
            pi_str = _mp.nstr(_mp.pi, num_digits + 2)
            digits_str = pi_str.split(".")[1][:num_digits]
            digits = [int(ch) for ch in digits_str]
            if len(digits) == num_digits:
                print(f"[INFO] π を {num_digits} 桁計算しました (mpmath)")
                return digits
            print("[WARN] mpmath の出力桁数が不足。フォールバックします。",
                  file=sys.stderr)
        except Exception as e:  # pragma: no cover - 環境依存
            print(f"[WARN] mpmath による π 計算に失敗: {e}", file=sys.stderr)
    else:
        print("[WARN] mpmath が無いためハードコード済み π 桁を使用します。",
              file=sys.stderr)
    return _fallback_pi_digits(num_digits)


def _fallback_pi_digits(num_digits: int = NUM_DIGITS) -> list[int]:
    """mpmath が使えない場合の π 桁（1000桁を実データとし、不足分は巡回補完）。"""
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
    base = [int(ch) for ch in pi_1000]
    digits: list[int] = []
    while len(digits) < num_digits:
        digits.extend(base[: min(num_digits - len(digits), len(base))])
    return digits[:num_digits]


# ============================================================
# 2. ノイズ源の生成（5条件）
# ============================================================

def compute_inverse_pi(pi_digits: list[int]) -> list[int]:
    """逆π: d→10-d, d==0→10。値域 1〜10（0を含まない）。"""
    return [10 - d if d != 0 else 10 for d in pi_digits]


def compute_inverse_pi_mean_matched(
    pi_digits: list[int],
    target_mean: float | None = None,
) -> tuple[list[float], float]:
    """
    逆πを線形スケールし、平均ステップ幅を π の非ゼロ桁平均に一致させる。
    戻り値: (スケール済み桁列, 適用したスケール係数)
    """
    if target_mean is None:
        nonzero = [d for d in pi_digits if d != 0]
        target_mean = float(np.mean(nonzero)) if nonzero else 5.0
    inv = compute_inverse_pi(pi_digits)
    inv_mean = float(np.mean(inv))
    scale = target_mean / inv_mean if inv_mean > 0 else 1.0
    return [d * scale for d in inv], scale


def compute_inverse_pi_width_matched(
    pi_digits: list[int],
) -> tuple[list[float], float]:
    """
    逆πを線形スケールし、『全ステップの平均歩幅』を π の平均（0を含む）に一致させる。

    inverse_pi_mean_matched が『動いた時の歩幅』を揃えるのに対し、これは
    digit=0 の停止分まで含めた総移動量を揃える。両方あると
      ・歩幅の大きさ     → width_matched で統制
      ・1回の移動量      → mean_matched で統制
      ・0による静止そのもの → 両者の差
    を分離できる。既定のマトリクスには含めず、--with-width-matched で有効化する。
    """
    target_mean = float(np.mean(pi_digits)) if pi_digits else 4.5
    inv = compute_inverse_pi(pi_digits)
    inv_mean = float(np.mean(inv))
    scale = target_mean / inv_mean if inv_mean > 0 else 1.0
    return [d * scale for d in inv], scale


def compute_uniform_nonzero(
    n: int,
    low: int = 1,
    high: int = 10,
    seed: int = SEED,
) -> list[int]:
    """一様整数ノイズ 1〜10（0を含まない対照条件）。"""
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(low, high + 1, size=n)]


def compute_pi_nonzero_resampled(
    pi_digits: list[int],
    seed: int = SEED,
) -> list[int]:
    """πの0をπの非ゼロ桁から復元抽出した値で置換する。長さは同一で0なし。"""
    arr = np.array(pi_digits, dtype=np.int64)
    nonzero = arr[arr != 0]
    if nonzero.size == 0:
        return [int(v) for v in arr]
    zero_mask = arr == 0
    n_zero = int(zero_mask.sum())
    if n_zero > 0:
        rng = np.random.default_rng(seed)
        arr = arr.copy()
        arr[zero_mask] = rng.choice(nonzero, size=n_zero, replace=True)
    return [int(v) for v in arr]


def build_noise_sources(pi_digits: list[int]) -> tuple[dict[str, list], dict[str, float]]:
    """5条件（＋任意の第2統制）すべてのノイズ列を生成する。"""
    inv_pi = compute_inverse_pi(pi_digits)
    inv_matched, scale = compute_inverse_pi_mean_matched(pi_digits)
    inv_width, width_scale = compute_inverse_pi_width_matched(pi_digits)
    uniform = compute_uniform_nonzero(len(pi_digits), seed=SEED)
    pi_resampled = compute_pi_nonzero_resampled(pi_digits, seed=SEED + 1)

    sources: dict[str, list] = {
        "pi": pi_digits,
        "inverse_pi": inv_pi,
        "inverse_pi_mean_matched": inv_matched,
        "uniform_nonzero": uniform,
        "pi_nonzero_resampled": pi_resampled,
        "inverse_pi_width_matched": inv_width,
    }

    meta: dict[str, float] = {
        "inverse_pi_mean_matched_scale": scale,
        "inverse_pi_width_matched_scale": width_scale,
    }
    for name, seq in sources.items():
        arr = np.asarray(seq, dtype=float)
        meta[f"{name}_mean"] = float(arr.mean())
        meta[f"{name}_min"] = float(arr.min())
        meta[f"{name}_max"] = float(arr.max())
        meta[f"{name}_zero_rate"] = float(np.mean(arr == 0.0))
    return sources, meta


# ============================================================
# 3. 目的関数（6種・すべて大域最適解が既知）
# ============================================================

def rastrigin_1d(x):
    """1D Rastrigin 関数（既存 API 互換: スカラー/配列の両方を受け取る）。"""
    arr = np.asarray(x, dtype=float)
    return arr ** 2 + 10.0 * (1.0 - np.cos(2.0 * np.pi * arr))


def _obj_rastrigin(X: np.ndarray) -> np.ndarray:
    """Rastrigin（n次元・等間隔の多峰性）。"""
    return np.sum(X * X + 10.0 * (1.0 - np.cos(2.0 * np.pi * X)), axis=1)


def _obj_ackley(X: np.ndarray) -> np.ndarray:
    """Ackley（広いほぼ平坦な領域＋鋭い大域最小）。"""
    d = X.shape[1]
    mean_sq = np.sum(X * X, axis=1) / d
    mean_cos = np.sum(np.cos(2.0 * np.pi * X), axis=1) / d
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(np.maximum(mean_sq, 0.0)))
        - np.exp(mean_cos)
        + 20.0
        + math.e
    )


def _obj_griewank(X: np.ndarray) -> np.ndarray:
    """Griewank（緩い二次項に周期的な谷が乗る形状）。"""
    d = X.shape[1]
    denom = np.sqrt(np.arange(1, d + 1, dtype=float))
    sum_term = np.sum(X * X, axis=1) / 4000.0
    prod_term = np.prod(np.cos(X / denom), axis=1)
    return sum_term - prod_term + 1.0


def _obj_rosenbrock(X: np.ndarray) -> np.ndarray:
    """Rosenbrock（狭いバナナ谷。大域最適解は (1,1)）。"""
    a = X[:, :-1]
    b = X[:, 1:]
    return np.sum(100.0 * (b - a * a) ** 2 + (1.0 - a) ** 2, axis=1)


def _obj_sphere(X: np.ndarray) -> np.ndarray:
    """Sphere（単一の谷だけの基準線）。"""
    return np.sum(X * X, axis=1)


@dataclass(frozen=True)
class ObjectiveSpec:
    """目的関数仕様。成功判定は『既知の大域最適解からの距離 < tol』で統一する。"""

    key: str
    label: str
    dim: int
    x_opt: tuple[float, ...]
    init_bound: float
    tol: float
    shape_note: str
    func: Callable[[np.ndarray], np.ndarray]


OBJECTIVES: list[ObjectiveSpec] = [
    ObjectiveSpec("rastrigin_1d", "Rastrigin 1D", 1, (0.0,), 4.0, 0.5,
                  "既存の主ベンチマーク（等間隔の多峰性）", _obj_rastrigin),
    ObjectiveSpec("rastrigin_2d", "Rastrigin 2D", 2, (0.0, 0.0), 4.0, 0.5,
                  "多峰性の次元上げ（谷の組み合わせ爆発）", _obj_rastrigin),
    ObjectiveSpec("ackley", "Ackley 2D", 2, (0.0, 0.0), 4.0, 0.5,
                  "広い平坦部＋鋭い大域最小", _obj_ackley),
    ObjectiveSpec("griewank", "Griewank 2D", 2, (0.0, 0.0), 6.0, 0.5,
                  "緩い二次項＋周期谷（遠い初期点でも届くか）", _obj_griewank),
    ObjectiveSpec("rosenbrock", "Rosenbrock 2D", 2, (1.0, 1.0), 3.0, 0.5,
                  "狭いバナナ谷（方向の整合が要る）", _obj_rosenbrock),
    ObjectiveSpec("sphere", "Sphere 2D", 2, (0.0, 0.0), 5.0, 0.5,
                  "単一谷の基準線（脱出性能の見かけを測る）", _obj_sphere),
]

OBJECTIVE_INDEX: dict[str, int] = {spec.key: i for i, spec in enumerate(OBJECTIVES)}
NOISE_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(NOISE_TYPES + EXTRA_NOISE_TYPES)
}
MATRIX_T0_INDEX: dict[float, int] = {v: i for i, v in enumerate(MATRIX_T0_LIST)}
MATRIX_COOLING_INDEX: dict[float, int] = {v: i for i, v in enumerate(MATRIX_COOLING_LIST)}


def get_objective(key: str) -> ObjectiveSpec:
    """キーで目的関数仕様を取り出す（未知キーは黙って黙殺しない）。"""
    try:
        return OBJECTIVES[OBJECTIVE_INDEX[key]]
    except KeyError:
        known = ", ".join(OBJECTIVE_INDEX)
        raise KeyError(f"未知の目的関数です: {key}（既知: {known}）") from None


def objective_distance(spec: ObjectiveSpec, X: np.ndarray) -> np.ndarray:
    """大域最適解からのユークリッド距離（試行分の配列を返す）。"""
    arr = np.atleast_2d(np.asarray(X, dtype=float))
    x_opt = np.asarray(spec.x_opt, dtype=float)
    return np.linalg.norm(arr - x_opt, axis=1)


# ============================================================
# 4. Simulated Annealing（スカラー版・後方互換）
# ============================================================

@dataclass
class SATrialResult:
    """1試行分の SA 実行結果と追跡メトリクス。

    旧版はタプル `(trajectory, final_x, accept_rate)` を返していた。
    __iter__ を実装しているので、旧呼び出し側のアンパックはそのまま動く。
    """

    trajectory: np.ndarray          # 各ステップ後の位置（長さ steps+1）
    final_x: float                  # 1D は最終 x、多次元は最適解からの距離
    accept_rate: float              # 受理率 (0〜1)
    step_widths: np.ndarray         # 各ステップの |step|（長さ steps）
    zero_steps: int                 # |step| == 0 のステップ数
    steps_to_threshold: int | None  # 閾値内へ初到達したステップ（未達は None）
    mean_step_width: float
    max_step_width: float
    final_value: float = float("nan")     # 最終目的関数値
    final_distance: float = float("nan")  # 大域最適解からの距離

    def __iter__(self):
        yield self.trajectory
        yield self.final_x
        yield self.accept_rate


class _LegacyRandomShim:
    """np.random グローバル関数の薄い互換レイヤ（旧 main の再現用）。"""

    def random(self, size=None):
        return np.random.random() if size is None else np.random.random(size)

    def uniform(self, low, high=None, size=None):
        if high is None:
            low, high = 0.0, low
        if size is None:
            return np.random.uniform(low, high)
        return np.random.uniform(low, high, size)

    def standard_normal(self, size=None):
        return (np.random.standard_normal() if size is None
                else np.random.standard_normal(size))


def simulated_annealing(
    noise_digits: list,
    T0: float = T0,
    cooling: float = COOLING,
    steps: int = N_STEPS,
    step_scale: float = STEP_SCALE,
    threshold: float | None = THRESHOLD,
    x_init: float | None = None,
    rng=None,
    objective: ObjectiveSpec | None = None,
    seed_entropy: int | None = None,
) -> SATrialResult:
    """
    Simulated Annealing を 1 試行実行する（スカラー版・可視化デモ向け）。

    ステップ幅 = digit × step_scale × direction × (T/T0 + 0.01)

    - objective 省略時は 1D Rastrigin（旧版と同じ挙動）。
    - rng 省略時は np.random グローバル状態を使用（旧版の
      `np.random.seed(SEED + trial)` パターン互換）。新コードでは
      seed_entropy か rng を明示し、条件をまたいだ同一シード再利用を避ける。
    - digit=0 のとき step=0（完全停止）。π 条件で約10%発生する。
    """
    if objective is None:
        objective = get_objective("rastrigin_1d")
    if rng is None:
        if seed_entropy is not None:
            rng = np.random.default_rng(np.random.SeedSequence(entropy=seed_entropy))
        else:
            rng = _LegacyRandomShim()

    dim = objective.dim
    x_opt = np.asarray(objective.x_opt, dtype=float)
    is_1d = dim == 1

    if x_init is None:
        x = np.atleast_1d(
            np.asarray(rng.uniform(-objective.init_bound, objective.init_bound, size=dim), dtype=float)
        )
    else:
        x = np.atleast_1d(np.asarray(x_init, dtype=float))
        if x.size != dim:
            raise ValueError(f"x_init の次元数が一致しません: {x.size} != {dim}")

    T = float(T0)
    trajectory = np.empty((steps + 1, 1) if not is_1d else (steps + 1,), dtype=float)
    trajectory[0] = x[0] if is_1d else x
    step_widths = np.empty(steps, dtype=float)
    f_x = float(objective.func(x.reshape(1, dim))[0])
    accepts = 0
    zero_steps = 0
    dist = float(np.linalg.norm(x - x_opt))
    steps_to_threshold: int | None = (
        0 if (threshold is not None and dist < threshold) else None
    )
    n_digits = len(noise_digits)

    for i in range(steps):
        digit = float(noise_digits[i % n_digits])

        direction = np.atleast_1d(np.asarray(rng.standard_normal(dim), dtype=float))
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            direction = np.zeros(dim)
            direction[0] = 1.0
        else:
            direction = direction / norm

        step_vec = digit * step_scale * direction * (T / T0 + 0.01)
        abs_step = float(np.linalg.norm(step_vec))
        step_widths[i] = abs_step
        if abs_step == 0.0:
            zero_steps += 1

        x_new = x + step_vec
        f_new = float(objective.func(x_new.reshape(1, dim))[0])
        delta = f_new - f_x

        if delta < 0 or float(rng.random()) < math.exp(-delta / max(T, 0.001)):
            x = x_new
            f_x = f_new
            accepts += 1

        T = max(T * cooling, 0.001)
        trajectory[i + 1] = x[0] if is_1d else x

        if steps_to_threshold is None and threshold is not None:
            if float(np.linalg.norm(x - x_opt)) < threshold:
                steps_to_threshold = i + 1

    return SATrialResult(
        trajectory=trajectory,
        final_x=float(x[0]) if is_1d else float(np.linalg.norm(x - x_opt)),
        accept_rate=accepts / steps if steps > 0 else 0.0,
        step_widths=step_widths,
        zero_steps=zero_steps,
        steps_to_threshold=steps_to_threshold,
        mean_step_width=float(step_widths.mean()) if steps > 0 else 0.0,
        max_step_width=float(step_widths.max()) if steps > 0 else 0.0,
        final_value=f_x,
        final_distance=float(np.linalg.norm(x - x_opt)),
    )


# ============================================================
# 5. マトリクス実験エンジン（試行軸を numpy でベクトル化）
# ============================================================

def temperature_schedule(steps: int, T0_value: float, cooling: float) -> np.ndarray:
    """各ステップの温度 T_i = max(T0 * cooling^i, 0.001)（スカラー版と同定義）。"""
    idx = np.arange(steps, dtype=float)
    return np.maximum(float(T0_value) * np.power(float(cooling), idx), 0.001)


def step_width_sequence(
    digits: Iterable[float],
    steps: int,
    T0_value: float,
    cooling: float,
    step_scale: float = STEP_SCALE,
) -> np.ndarray:
    """
    受理結果に依存しない『提案ステップの大きさ』列を先計算する。

    |step_i| = digit_i × step_scale × (T_i / T0 + 0.01)
    デジタル列はステップ番号だけで決まるため、ステップ幅ヒストグラムや
    ゼロステップ率は SA ループを回さずに正確に求まる。
    """
    temps = temperature_schedule(steps, T0_value, cooling)
    factors = temps / float(T0_value) + 0.01
    tiled = np.resize(np.asarray(list(digits), dtype=float), steps)
    return tiled * float(step_scale) * factors


def derive_cell_rng(
    seed_entropy: int,
    objective_key: str,
    T0_value: float,
    cooling: float,
    noise_name: str,
) -> np.random.Generator:
    """
    セル専用に独立した乱数ストリームを作る。

    SeedSequence の spawn_key に (目的関数, T0, 冷却率, ノイズ条件) の添字を使う。
    条件をまたいで同一シードを再設定しないため、試行番号をまたいだ
    見かけのペア化（疑似ペア化）は発生しない。
    """
    spawn_key = (
        OBJECTIVE_INDEX[objective_key],
        MATRIX_T0_INDEX[float(T0_value)],
        MATRIX_COOLING_INDEX[float(cooling)],
        NOISE_INDEX[noise_name],
    )
    return np.random.default_rng(
        np.random.SeedSequence(entropy=int(seed_entropy), spawn_key=spawn_key)
    )


@dataclass
class CellOutcome:
    """1セル（ノイズ条件 × 目的関数 × 温度帯 × 冷却率）の実行結果。"""

    noise: str
    objective_key: str
    T0_value: float
    cooling: float
    steps: int
    n_trials: int
    successes: np.ndarray       # bool: 最終位置が tol 以内（旧版定義と一致・主指標）
    arrivals: np.ndarray        # bool: 一度でも tol に入った（到達経験）
    first_arrival: np.ndarray   # int: 初到達ステップ（未達 -1）
    maes: np.ndarray            # 最終位置の成分平均絶対誤差
    final_dists: np.ndarray     # 最適解からのユークリッド距離
    final_values: np.ndarray    # 最終目的関数値
    accept_rates: np.ndarray    # 受理率（0幅ステップは提案されないので分母に含めない）
    curve_steps: list[int]
    curve_best: list[float]     # 過去最良距離の試行平均
    curve_current: list[float]  # 現在距離の試行平均
    step_widths: np.ndarray     # 長さ steps の |提案幅|（全試行共通）
    traj0: np.ndarray | None    # 1D のとき試行0の軌跡（可視化用）
    seed_entropy: int

    def cell_id(self) -> str:
        return (
            f"{self.objective_key}|T0={self.T0_value:g}"
            f"|cool={self.cooling:g}|{self.noise}"
        )

    def stats(self) -> dict:
        n = self.n_trials
        k = int(self.successes.sum())
        reached = self.first_arrival[self.first_arrival >= 0]
        return {
            "noise": self.noise,
            "objective": self.objective_key,
            "T0": self.T0_value,
            "cooling": self.cooling,
            "n_trials": n,
            "n_steps": self.steps,
            "n_success": k,
            "success_rate": k / n if n else float("nan"),
            "n_arrival": int(self.arrivals.sum()),
            "arrival_rate": float(self.arrivals.sum()) / n if n else float("nan"),
            "mae_mean": float(self.maes.mean()) if n else float("nan"),
            "mae_median": float(np.median(self.maes)) if n else float("nan"),
            "final_dist_mean": float(self.final_dists.mean()) if n else float("nan"),
            "final_dist_median": float(np.median(self.final_dists)) if n else float("nan"),
            "final_value_mean": float(self.final_values.mean()) if n else float("nan"),
            "final_value_median": float(np.median(self.final_values)) if n else float("nan"),
            "accept_rate_mean": float(self.accept_rates.mean()) if n else float("nan"),
            "zero_step_rate": float(np.mean(self.step_widths == 0.0)),
            "mean_step_width": float(self.step_widths.mean()),
            "max_step_width": float(self.step_widths.max()),
            "n_reached": int(reached.size),
            "escape_step_median": float(np.median(reached)) if reached.size else float("nan"),
            "escape_step_mean": float(reached.mean()) if reached.size else float("nan"),
            "seed_entropy": self.seed_entropy,
        }


def run_sa_cell(
    digits: list | np.ndarray,
    objective: ObjectiveSpec,
    T0_value: float,
    cooling: float,
    noise_name: str,
    steps: int = N_STEPS,
    n_trials: int = MATRIX_TRIALS,
    step_scale: float = STEP_SCALE,
    seed_entropy: int = SEED,
    curve_stride: int = CURVE_STRIDE,
) -> CellOutcome:
    """
    1セルの SA を全試行ベクトル化して実行する。

    各ステップ:
      1. digit=0 なら完全停止（提案しない＝位置も目的関数値も不変）
      2. そうでなければ単位方向（多次元は正規乱数を正規化）に幅 w を乗じて提案
      3. Met規則 exp(-delta/T) で受理（delta<0 は常に受理）
    """
    rng = derive_cell_rng(seed_entropy, objective.key, T0_value, cooling, noise_name)
    dim = objective.dim
    x_opt = np.asarray(objective.x_opt, dtype=float)
    tol = objective.tol
    n = int(n_trials)

    temps = temperature_schedule(steps, T0_value, cooling)
    widths = step_width_sequence(digits, steps, T0_value, cooling, step_scale)
    is_1d = dim == 1

    x = rng.uniform(-objective.init_bound, objective.init_bound, size=(n, dim))
    f = objective.func(x)
    dist = np.linalg.norm(x - x_opt, axis=1)
    best = dist.copy()
    reached = dist < tol
    first = np.where(reached, 0, -1).astype(np.int64)
    accepts = np.zeros(n, dtype=np.int64)
    moved = np.zeros(n, dtype=np.int64)  # 実際に提案が行われたステップ数

    traj0 = np.empty(steps + 1, dtype=float) if is_1d else None
    if traj0 is not None:
        traj0[0] = x[0, 0]

    curve_steps: list[int] = []
    curve_best: list[float] = []
    curve_current: list[float] = []

    for i in range(steps):
        w = float(widths[i])
        if w > 0.0:
            dirs = rng.standard_normal((n, dim))
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0  # まれなゼロノルムで nan を出さない
            x_new = x + w * (dirs / norms)
            f_new = objective.func(x_new)
            delta = f_new - f
            p_accept = np.exp(-np.clip(delta, 0.0, 700.0) / max(float(temps[i]), 0.001))
            take = (delta < 0.0) | (rng.random(n) < p_accept)
            if take.any():
                x = np.where(take[:, None], x_new, x)
                f = np.where(take, f_new, f)
                accepts += take
            moved += 1

        dist = np.linalg.norm(x - x_opt, axis=1)
        np.minimum(best, dist, out=best)
        hit = (~reached) & (dist < tol)
        if hit.any():
            first[hit] = i + 1
            reached |= hit

        if (i % curve_stride == 0) or (i == steps - 1):
            curve_steps.append(i + 1)
            curve_best.append(float(best.mean()))
            curve_current.append(float(dist.mean()))

        if traj0 is not None:
            traj0[i + 1] = x[0, 0]

    maes = np.mean(np.abs(x - x_opt), axis=1) if not is_1d else np.abs(x[:, 0] - x_opt[0])
    denom = np.maximum(moved, 1)  # 受理率は『提案できたステップ』を分母にする
    final_dists = np.linalg.norm(x - x_opt, axis=1)

    return CellOutcome(
        noise=noise_name,
        objective_key=objective.key,
        T0_value=float(T0_value),
        cooling=float(cooling),
        steps=int(steps),
        n_trials=n,
        successes=final_dists < tol,   # 主指標: 終わって tol 以内に留まったか
        arrivals=reached,              # 補助: 一度でも tol に入ったか
        first_arrival=first,
        maes=maes,
        final_dists=final_dists,
        final_values=objective.func(x),
        accept_rates=accepts / denom,
        curve_steps=curve_steps,
        curve_best=curve_best,
        curve_current=curve_current,
        step_widths=widths,
        traj0=traj0,
        seed_entropy=int(seed_entropy),
    )


# ============================================================
# 6. 統計ヘルパー（scipy 非依存・正規近似）
# ============================================================

def _norm_cdf(x: float) -> float:
    """標準正規分布 CDF（erf ベース。scipy 不要）。"""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """成功率の Wilson 信頼区間（割合が 0/1 でも崩れない）。"""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0))
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_score_diff_ci(
    k1: int, n1: int, k2: int, n2: int, z: float = Z_95
) -> tuple[float, float, float]:
    """2比例の差（リスク差）の Newcombe 型信頼区間（Wilson を合成）。"""
    if n1 <= 0 or n2 <= 0:
        return (float("nan"), float("nan"), float("nan"))
    lo1, hi1 = wilson_interval(k1, n1, z)
    lo2, hi2 = wilson_interval(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    diff = p1 - p2
    lower = diff - ((p1 - lo1) + (hi2 - p2))
    upper = diff + ((hi1 - p1) + (p2 - lo2))
    return (diff, max(-1.0, lower), min(1.0, upper))


def odds_ratio_ci(
    k1: int, n1: int, k2: int, n2: int, z: float = Z_95
) -> tuple[float, float, float]:
    """オッズ比と対数法の信頼区間。ゼロセルは Haldane(0.5) 補正で安定化。"""
    if n1 <= 0 or n2 <= 0:
        return (float("nan"), float("nan"), float("nan"))
    a, b = k1 + 0.5, (n1 - k1) + 0.5
    c, d = k2 + 0.5, (n2 - k2) + 0.5
    orr = (a * d) / (b * c)
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    return (orr, orr * math.exp(-z * se), orr * math.exp(z * se))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """同順位を平均する順位付け（scipy.stats.rankdata の最小代替）。"""
    arr = np.asarray(x, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=float)
    srt = arr[order]
    i = 0
    while i < arr.size:
        j = i
        while j + 1 < arr.size and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def two_proportion_permutation_p(
    s1: np.ndarray,
    s2: np.ndarray,
    n_perm: int = N_PERM,
    rng: np.random.Generator | None = None,
) -> float:
    """
    2つの成功ラベル列の差を、ラベル置換で検定する（両側・非対応）。
    正規近似に頼らず実データで p 値を出す。返りは (ge+1)/(reps+1) の保守型。
    """
    a = np.asarray(s1, dtype=np.float64)
    b = np.asarray(s2, dtype=np.float64)
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = np.concatenate([a, b])
    total = float(pooled.sum())
    obs_abs = abs(float(a.mean()) - float(b.mean()))
    if rng is None:
        rng = np.random.default_rng(
            np.random.SeedSequence(entropy=SEED, spawn_key=(0, 0, 0, 0))
        )
    n_total = pooled.size
    batch = 1000
    ge = 0
    reps = 0
    for _ in range(max(1, int(n_perm) // batch)):
        tile = np.tile(pooled, (batch, 1))
        perm = rng.permuted(tile, axis=1)
        sums1 = perm[:, :n1].sum(axis=1)
        diffs = sums1 / n1 - (total - sums1) / n2
        ge += int(np.count_nonzero(np.abs(diffs) >= obs_abs - 1e-12))
        reps += batch
    return (ge + 1) / (reps + 1)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """小標本補正付き効果量 Hedges' g（a - b 方向）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    s_pool = math.sqrt(
        ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    )
    if s_pool == 0.0:
        return float("nan")
    d = (a.mean() - b.mean()) / s_pool
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(d * correction)


def welch_mean_diff(a: np.ndarray, b: np.ndarray, z: float = Z_95) -> dict[str, float]:
    """対応のない平均差（Welch 型 SE・正規近似 p・Hedges' g）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = a.size, b.size
    out: dict[str, float] = {
        "mean_a": float(a.mean()) if n1 else float("nan"),
        "mean_b": float(b.mean()) if n2 else float("nan"),
        "diff": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "p_value": float("nan"),
        "hedges_g": float("nan"),
    }
    if n1 < 2 or n2 < 2:
        return out
    var = a.var(ddof=1) / n1 + b.var(ddof=1) / n2
    se = math.sqrt(var) if var > 0 else 0.0
    diff = float(a.mean() - b.mean())
    out["diff"] = diff
    out["hedges_g"] = hedges_g(a, b)
    if se == 0.0:
        return out
    out["ci_lo"] = diff - z * se
    out["ci_hi"] = diff + z * se
    out["p_value"] = min(1.0, 2.0 * (1.0 - _norm_cdf(abs(diff / se))))
    return out


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Mann-Whitney U（並み・正規近似・連続性補正）。効果量は rank-biserial。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = a.size, b.size
    out: dict[str, float] = {
        "median_a": float(np.median(a)) if n1 else float("nan"),
        "median_b": float(np.median(b)) if n2 else float("nan"),
        "n_a": float(n1),
        "n_b": float(n2),
        "u_statistic": float("nan"),
        "rank_biserial": float("nan"),
        "p_value": float("nan"),
    }
    if n1 < 3 or n2 < 3:
        return out
    allv = np.concatenate([a, b])
    ranks = _rankdata(allv)
    u1 = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2.0)
    n_total = n1 + n2
    _, counts = np.unique(allv, return_counts=True)
    tie_term = float(np.sum(counts.astype(float) ** 3 - counts.astype(float)))
    var = (n1 * n2 / 12.0) * ((n_total + 1) - tie_term / (n_total * (n_total - 1)))
    sigma = math.sqrt(var) if var > 0 else 0.0
    out["u_statistic"] = u1
    out["rank_biserial"] = float(1.0 - 2.0 * u1 / (n1 * n2))
    if sigma == 0.0:
        return out
    z_stat = (abs(u1 - n1 * n2 / 2.0) - 0.5) / sigma
    out["p_value"] = min(1.0, 2.0 * (1.0 - _norm_cdf(max(z_stat, 0.0))))
    return out


def bonferroni_adjust(p: float, m: int) -> float:
    """Bonferroni 補正。NaN は NaN のまま返す（欠測を『有意でない』にしない）。"""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return float("nan")
    return min(1.0, float(p) * max(1, int(m)))


def compute_stats(final_x_list: list[float]) -> tuple[float, float, float]:
    """後方互換: 1D Rastrigin（大域最適解 x=0）の成功率%・平均誤差・中央誤差。"""
    arr = np.asarray(final_x_list, dtype=float)
    success_rate = float(np.mean(np.abs(arr) < THRESHOLD) * 100.0)
    mean_error = float(np.mean(np.abs(arr)))
    median_error = float(np.median(np.abs(arr)))
    return (success_rate, mean_error, median_error)


# ============================================================
# 7. マトリクス実験の駆動
# ============================================================

def cell_spawn_key(objective_key: str, T0_value: float, cooling: float, noise: str) -> tuple[int, int, int, int]:
    """再現監査用: このセルが使った乱数ストリームの座標を返す。"""
    return (
        OBJECTIVE_INDEX[objective_key],
        MATRIX_T0_INDEX[float(T0_value)],
        MATRIX_COOLING_INDEX[float(cooling)],
        NOISE_INDEX[noise],
    )


def run_matrix_experiment(
    n_trials: int = MATRIX_TRIALS,
    steps: int = N_STEPS,
    seed_entropy: int = SEED,
    step_scale: float = STEP_SCALE,
    curve_stride: int = CURVE_STRIDE,
    include_extra_noise: bool = False,
    progress: bool = True,
) -> dict:
    """
    5ノイズ条件 × 6目的関数 × 4温度帯 × 3冷却率 のマトリクスを実行する。

    戻り値（メモリ常駐）:
      cells        : list[CellOutcome]  試行単位の配列を含む（検定・図描画用）
      trial_rows   : list[dict]         試行生データ（CSV 出力用）
      comparisons  : list[dict]         ペアワイズ比較（Bonferroni 補正後つき）
      meta         : dict               設定・ノイズ統計・実行時間など
    """
    pi_digits = compute_pi_digits(NUM_DIGITS)
    sources, noise_meta = build_noise_sources(pi_digits)
    noises = active_noise_conditions(include_extra_noise)

    cells: list[CellOutcome] = []
    trial_rows: list[dict] = []
    total = len(noises) * len(OBJECTIVES) * len(MATRIX_T0_LIST) * len(MATRIX_COOLING_LIST)
    done = 0
    t_start = time.perf_counter()

    for spec in OBJECTIVES:
        for t0_value in MATRIX_T0_LIST:
            for cooling in MATRIX_COOLING_LIST:
                for noise in noises:
                    cell = run_sa_cell(
                        digits=sources[noise],
                        objective=spec,
                        T0_value=t0_value,
                        cooling=cooling,
                        noise_name=noise,
                        steps=steps,
                        n_trials=n_trials,
                        step_scale=step_scale,
                        seed_entropy=seed_entropy,
                        curve_stride=curve_stride,
                    )
                    cells.append(cell)
                    spawn = cell_spawn_key(spec.key, t0_value, cooling, noise)
                    for t in range(cell.n_trials):
                        trial_rows.append(
                            {
                                "cell_id": cell.cell_id(),
                                "objective": spec.key,
                                "noise": noise,
                                "T0": t0_value,
                                "cooling": cooling,
                                "trial": t,
                                "spawn_key": "|".join(str(v) for v in spawn),
                                "success": int(bool(cell.successes[t])),
                                "arrival": int(bool(cell.arrivals[t])),
                                "first_arrival_step": int(cell.first_arrival[t]),
                                "mae": float(cell.maes[t]),
                                "final_distance": float(cell.final_dists[t]),
                                "final_value": float(cell.final_values[t]),
                                "accept_rate": float(cell.accept_rates[t]),
                            }
                        )
                done += 1
                if progress and (done % 12 == 0 or done == total):
                    elapsed = time.perf_counter() - t_start
                    print(
                        f"      [{done:>3}/{total}] {spec.key:<12} T0={t0_value:<4g} "
                        f"cool={cooling:<6g} {elapsed:6.1f}s",
                        flush=True,
                    )

    comparisons = build_comparison_rows(cells, seed_entropy=seed_entropy)
    elapsed = time.perf_counter() - t_start

    meta = {
        "params": {
            "n_trials_per_cell": n_trials,
            "n_steps": steps,
            "seed_entropy": seed_entropy,
            "step_scale": step_scale,
            "curve_stride": curve_stride,
            "T0_list": list(MATRIX_T0_LIST),
            "cooling_list": list(MATRIX_COOLING_LIST),
            "noise_conditions": noises,
            "include_extra_noise": include_extra_noise,
            "objectives": [s.key for s in OBJECTIVES],
            "success_rule": "主指標 success = 最終位置が大域最適解から tol 以内（旧版定義と一致）。補助指標 arrival = 一度でも tol に入った",
            "rng_policy": (
                "SeedSequence(entropy=SEED, spawn_key=(objective,T0,cooling,noise)) "
                "でセルごとに独立ストリーム。同一シードの条件跨ぎ再利用はなし。"
            ),
            "accept_rate_definition": "提案が行われたステップのうち Met規則で受理された割合（digit=0 は分母に含めない）",
        },
        "pi_digits_source": "mpmath" if _HAS_MPMATH else "hardcoded_fallback",
        "pi_digits_used": NUM_DIGITS,
        "noise_meta": noise_meta,
        "elapsed_seconds": elapsed,
        "n_cells": len(cells),
        "n_trial_rows": len(trial_rows),
    }

    return {
        "cells": cells,
        "trial_rows": trial_rows,
        "comparisons": comparisons,
        "meta": meta,
        "noise_sources": sources,
    }


# ============================================================
# 8. ペアワイズ比較（事前に登録した 4 ペアのみ）
# ============================================================

COMPARISON_COLUMNS: list[str] = [
    "objective", "T0", "cooling", "cond_a", "cond_b", "metric",
    "n_a", "n_b", "value_a", "value_b", "diff", "ci_lo", "ci_hi",
    "odds_ratio", "or_ci_lo", "or_ci_hi",
    "effect_name", "effect", "p_value",
    "p_bonferroni_family", "p_bonferroni_global", "note",
]


def _comparison_note(metric: str, p_raw: float, p_bonf: float, diff: float, alpha: float = 0.05) -> str:
    """
    「有意」を断定しない書き分け。p 値の段階をそのまま文章の強さにする。
    diff は必ず a - b（成功率は高いほど良い、MAE と到達ステップ数は小さいほど良い）。
    """
    if p_raw is None or math.isnan(p_raw):
        return "検定不能（試行数が足りない）"
    if math.isnan(diff):
        direction = "差の向き不明"
    elif metric == "success_rate":
        direction = "a の成功率が高い" if diff > 0 else "b の成功率が高い"
    elif metric == "mae":
        direction = "a の誤差が小さい" if diff < 0 else "b の誤差が小さい"
    else:  # escape_steps
        direction = "a の方が早く到達" if diff < 0 else "b の方が早く到達"

    if not math.isnan(p_bonf) and p_bonf < alpha:
        return f"多重比較補正後も差が残る（{direction}）。歩幅統制だけでは説明しきれない可能性"
    if p_raw < alpha:
        return f"未補正では差があるが多重比較で消える（{direction}）。証拠としては弱い"
    return f"差は観察されない（{direction} 程度）。このセルでは平均ステップ幅説と矛盾しない"


def build_comparison_rows(
    cells: list[CellOutcome],
    seed_entropy: int = SEED,
    n_perm: int = N_PERM,
) -> list[dict]:
    """セル群から 4ペア × 3指標 の比較表を作る（すべて非対応検定）。"""
    groups: dict[tuple[str, float, float], dict[str, CellOutcome]] = {}
    for cell in cells:
        groups.setdefault((cell.objective_key, cell.T0_value, cell.cooling), {})[cell.noise] = cell

    perm_rng = np.random.default_rng(
        np.random.SeedSequence(entropy=seed_entropy, spawn_key=(99, 99, 99, 99))
    )
    rows: list[dict] = []

    for (obj_key, t0_value, cooling), grp in groups.items():
        base = {"objective": obj_key, "T0": t0_value, "cooling": cooling}

        for cond_a, cond_b in COMPARISON_PAIRS + EXTRA_COMPARISON_PAIRS:
            if cond_a not in grp or cond_b not in grp:
                continue
            ca, cb = grp[cond_a], grp[cond_b]

            # --- 1) 成功率（二値）: 差・Newcombe CI・オッズ比・置換検定 ---
            k1, n1 = int(ca.successes.sum()), ca.n_trials
            k2, n2 = int(cb.successes.sum()), cb.n_trials
            rd, rd_lo, rd_hi = newcombe_score_diff_ci(k1, n1, k2, n2)
            orr, or_lo, or_hi = odds_ratio_ci(k1, n1, k2, n2)
            p_perm = two_proportion_permutation_p(
                ca.successes.astype(float), cb.successes.astype(float), n_perm, perm_rng
            )
            wa = wilson_interval(k1, n1)
            wb = wilson_interval(k2, n2)
            rows.append({
                **base, "cond_a": cond_a, "cond_b": cond_b, "metric": "success_rate",
                "n_a": n1, "n_b": n2, "value_a": k1 / n1, "value_b": k2 / n2,
                "diff": rd, "ci_lo": rd_lo, "ci_hi": rd_hi,
                "odds_ratio": orr, "or_ci_lo": or_lo, "or_ci_hi": or_hi,
                "effect_name": "risk_diff", "effect": rd, "p_value": p_perm,
                "wilson_a_lo": wa[0], "wilson_a_hi": wa[1],
                "wilson_b_lo": wb[0], "wilson_b_hi": wb[1],
            })

            # --- 2) 平均絶対誤差（連続）: Welch 差・CI・Hedges' g ---
            wel = welch_mean_diff(ca.maes, cb.maes)
            rows.append({
                **base, "cond_a": cond_a, "cond_b": cond_b, "metric": "mae",
                "n_a": n1, "n_b": n2, "value_a": wel["mean_a"], "value_b": wel["mean_b"],
                "diff": wel["diff"], "ci_lo": wel["ci_lo"], "ci_hi": wel["ci_hi"],
                "odds_ratio": float("nan"), "or_ci_lo": float("nan"), "or_ci_hi": float("nan"),
                "effect_name": "hedges_g", "effect": wel["hedges_g"], "p_value": wel["p_value"],
            })

            # --- 3) 到達ステップ数（生存系の代わり・到達した試行のみ） ---
            arr_a = ca.first_arrival[ca.first_arrival >= 0].astype(float)
            arr_b = cb.first_arrival[cb.first_arrival >= 0].astype(float)
            mw = mann_whitney_u(arr_a, arr_b)
            med_diff = mw["median_a"] - mw["median_b"]
            rows.append({
                **base, "cond_a": cond_a, "cond_b": cond_b, "metric": "escape_steps",
                "n_a": int(arr_a.size), "n_b": int(arr_b.size),
                "value_a": mw["median_a"], "value_b": mw["median_b"],
                "diff": med_diff, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "odds_ratio": float("nan"), "or_ci_lo": float("nan"), "or_ci_hi": float("nan"),
                "effect_name": "rank_biserial", "effect": mw["rank_biserial"],
                "p_value": mw["p_value"],
            })

    # --- 多重比較補正（家族=指標ごと、大域=全比較） ---
    family_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row["metric"]] = family_counts.get(row["metric"], 0) + 1
    m_global = len(rows)

    for row in rows:
        m_family = family_counts[row["metric"]]
        row["p_bonferroni_family"] = bonferroni_adjust(row["p_value"], m_family)
        row["p_bonferroni_global"] = bonferroni_adjust(row["p_value"], m_global)
        row["note"] = _comparison_note(
            row["metric"], row["p_value"], row["p_bonferroni_global"], row["diff"]
        )
    return rows


# ============================================================
# 9. 出力（CSV / JSON）
# ============================================================

def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _json_safe(value):
    """NaN/Inf を JSON 許可値 (null) に変換。numpy 型は Python 本体へ。"""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    return value


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> int:
    if not rows:
        return 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if _is_nan(v) else v) for k, v in row.items()})
    return len(rows)


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def build_summary_rows(bundle: dict) -> list[dict]:
    """セル統計に Wilson 区間を添えてサマリ行を作る。"""
    rows = []
    for cell in bundle["cells"]:
        stat = cell.stats()
        k, n = stat["n_success"], stat["n_trials"]
        lo, hi = wilson_interval(k, n)
        stat["success_rate_ci_lo"] = lo
        stat["success_rate_ci_hi"] = hi
        stat["objective_label"] = get_objective(stat["objective"]).label
        stat["noise_label"] = NOISE_LABELS[stat["noise"]]
        rows.append(stat)
    return rows


def build_histogram_rows(bundle: dict, n_bins: int = 20) -> list[dict]:
    """最終位置（大域最適解からの距離）のヒストグラムデータを整える。"""
    rows = []
    for cell in bundle["cells"]:
        dists = cell.final_dists
        if dists.size == 0:
            continue
        counts, edges = np.histogram(dists, bins=n_bins, range=(0.0, max(float(dists.max()), 1e-9)))
        obj = get_objective(cell.objective_key)
        for i, count in enumerate(counts):
            rows.append({
                "cell_id": cell.cell_id(),
                "objective": cell.objective_key,
                "noise": cell.noise,
                "T0": cell.T0_value,
                "cooling": cell.cooling,
                "bin_center": 0.5 * (edges[i] + edges[i + 1]),
                "bin_lo": edges[i],
                "bin_hi": edges[i + 1],
                "count": int(count),
                "tol": obj.tol,
            })
    return rows


def summarize_step_width_hypothesis(bundle: dict) -> dict:
    """
    「逆πの差は平均ステップ幅で説明できるか」を数値で要約する。

    主眼: raw gap (pi vs inverse_pi) と matched gap (pi vs mean-matched) を比べ、
    歩幅統制で差がどれだけ縮んだかを見る。縮まないセルがどれかを条件付きで拾う。
    """
    rows = bundle["comparisons"]

    def _select(cond_a: str, cond_b: str, metric: str = "success_rate") -> list[dict]:
        return [
            r for r in rows
            if r["metric"] == metric and r["cond_a"] == cond_a and r["cond_b"] == cond_b
        ]

    def _block(sel: list[dict]) -> dict:
        if not sel:
            return {}
        diffs = np.array([r["diff"] for r in sel], dtype=float)
        return {
            "n_cells": len(sel),
            "mean_diff_a_minus_b": float(np.mean(diffs)),
            "median_abs_diff": float(np.median(np.abs(diffs))),
            "n_raw_p_lt_005": int(sum(1 for r in sel if not math.isnan(r["p_value"]) and r["p_value"] < 0.05)),
            "n_bonferroni_lt_005": int(sum(
                1 for r in sel
                if not math.isnan(r["p_bonferroni_global"]) and r["p_bonferroni_global"] < 0.05
            )),
        }

    raw = _select("pi", "inverse_pi")
    matched = _select("pi", "inverse_pi_mean_matched")
    residual = _select("inverse_pi", "inverse_pi_mean_matched")

    raw_mean = float(np.mean([r["diff"] for r in raw])) if raw else float("nan")
    matched_mean = float(np.mean([r["diff"] for r in matched])) if matched else float("nan")
    explained = float("nan")
    if raw and matched and math.isfinite(raw_mean) and abs(raw_mean) > 1e-9:
        explained = float(1.0 - matched_mean / raw_mean)

    leftovers = sorted(
        [r for r in matched if not math.isnan(r["p_value"])],
        key=lambda r: r["p_value"],
    )[:8]

    return {
        "pi_vs_inverse_pi": _block(raw),
        "pi_vs_inverse_pi_mean_matched": _block(matched),
        "inverse_pi_vs_inverse_pi_mean_matched": _block(residual),
        "success_rate_gap_explained_ratio": explained,
        "strongest_pi_vs_mean_matched_cells": [
            {
                "objective": r["objective"], "T0": r["T0"], "cooling": r["cooling"],
                "risk_diff": r["diff"], "p_value": r["p_value"],
                "p_bonferroni_global": r["p_bonferroni_global"],
            }
            for r in leftovers
        ],
        "reading_guide": (
            "explained_ratio が 1 に近いほど『歩幅を揃えると差が消える』= 平均ステップ幅説を支持。"
            " 0 前後なら歩幅を揃えても差が残る= 条件付きで別要因の可能性。"
            " ただし本実験は 1セル 60試行の探索的検証であり、この数値は結論ではなく指標。"
        ),
    }


TRIAL_FIELDS: list[str] = [
    "cell_id", "objective", "noise", "T0", "cooling", "trial", "spawn_key",
    "success", "arrival", "first_arrival_step", "mae", "final_distance",
    "final_value", "accept_rate",
]

SUMMARY_FIELDS: list[str] = [
    "noise", "objective", "T0", "cooling", "n_trials", "n_steps",
    "n_success", "success_rate", "success_rate_ci_lo", "success_rate_ci_hi",
    "n_arrival", "arrival_rate",
    "mae_mean", "mae_median", "final_dist_mean", "final_dist_median",
    "final_value_mean", "final_value_median", "accept_rate_mean",
    "zero_step_rate", "mean_step_width", "max_step_width",
    "n_reached", "escape_step_median", "escape_step_mean", "seed_entropy",
]

COMPARISON_FIELDS: list[str] = COMPARISON_COLUMNS + [
    "wilson_a_lo", "wilson_a_hi", "wilson_b_lo", "wilson_b_hi",
]


def aggregate_cells(bundle: dict) -> list[dict]:
    """ノイズ条件 × 目的関数で温度帯を横断集計する（温度効きを見るときは summary CSV を使う）。"""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for cell in bundle["cells"]:
        buckets.setdefault((cell.objective_key, cell.noise), []).append(cell.stats())

    rows = []
    for (obj_key, noise), stats_list in buckets.items():
        rows.append({
            "objective": obj_key,
            "noise": noise,
            "n_temperature_cells": len(stats_list),
            "success_rate_mean": float(np.mean([s["success_rate"] for s in stats_list])),
            "success_rate_min": float(np.min([s["success_rate"] for s in stats_list])),
            "success_rate_max": float(np.max([s["success_rate"] for s in stats_list])),
            "mae_mean": float(np.mean([s["mae_mean"] for s in stats_list])),
            "accept_rate_mean": float(np.mean([s["accept_rate_mean"] for s in stats_list])),
            "mean_step_width": float(np.mean([s["mean_step_width"] for s in stats_list])),
            "zero_step_rate": float(np.mean([s["zero_step_rate"] for s in stats_list])),
            "escape_step_median": float(np.nanmean([s["escape_step_median"] for s in stats_list])),
        })
    return rows


def build_interpretation_notes(bundle: dict) -> list[str]:
    """実測値を踏まえた『言えること／言えないこと』。断定表現は使わない。"""
    hypo = summarize_step_width_hypothesis(bundle)
    explained = hypo.get("success_rate_gap_explained_ratio")
    raw = hypo.get("pi_vs_inverse_pi", {})
    matched = hypo.get("pi_vs_inverse_pi_mean_matched", {})
    notes: list[str] = []

    raw_gap = raw.get("mean_diff_a_minus_b", float("nan"))
    matched_gap = matched.get("mean_diff_a_minus_b", float("nan"))

    if math.isfinite(raw_gap) and math.isfinite(matched_gap):
        shrink = abs(raw_gap) - abs(matched_gap)
        notes.append(
            f"全 {raw.get('n_cells', 0)} セルの平均で、pi に対する成功率差は "
            f"逆π (inverse_pi) が {raw_gap:+.4f}、歩幅を揃えた条件 "
            f"(inverse_pi_mean_matched) が {matched_gap:+.4f} でした（絶対値の縮み {shrink:+.4f}）。"
        )
        if isinstance(explained, float) and math.isfinite(explained) and abs(raw_gap) >= 0.02:
            notes.append(
                f"この縮みは元の差の約 {explained * 100:.0f}% に相当します。ただし元の差が "
                f"2 ポイント未満のセルではこの割合は不安定なので、目安に留めてください。"
            )
        else:
            notes.append(
                "元々の平均差が小さいため『説明率』はあえて算出していません（差が無いものに比率は無い）。"
            )
    notes.append(
        f"全 {raw.get('n_cells', 0)} セル中、Bonferroni 補正後に成功率差が 0.05 を下回ったのは "
        f"pi vs inverse_pi で {raw.get('n_bonferroni_lt_005', 0)} セル、"
        f"pi vs mean-matched で {matched.get('n_bonferroni_lt_005', 0)} セルでした。"
    )
    notes.append(
        "温度帯と目的関数によって差の向きが変わるセルがあるため、『逆πが効く／効かない』を"
        "単一の結論に潰すことはできません。条件付きの記述にとどめます。"
    )
    notes.append(
        "本実験は 1セル 60試行・同一 π 桁列という探索的な規模です。p 値と信頼区間は"
        "『どの程度の一貫性があるか』の目安であり、機構の証明ではありません。"
    )
    return notes


def write_outputs(bundle: dict) -> dict:
    """results/ に CSV と JSON を書き出し、書行数を返す。"""
    ensure_output_dirs()
    counts = {}
    counts["optimization_results.csv"] = _write_csv(
        RESULTS_DIR / "optimization_results.csv", bundle["trial_rows"], TRIAL_FIELDS
    )
    summary_rows = build_summary_rows(bundle)
    counts["optimization_summary.csv"] = _write_csv(
        RESULTS_DIR / "optimization_summary.csv", summary_rows, SUMMARY_FIELDS
    )
    counts["comparisons.csv"] = _write_csv(
        RESULTS_DIR / "comparisons.csv", bundle["comparisons"], COMPARISON_FIELDS
    )
    hist_rows = build_histogram_rows(bundle)
    counts["final_position_histogram.csv"] = _write_csv(
        RESULTS_DIR / "final_position_histogram.csv",
        hist_rows,
        ["cell_id", "objective", "noise", "T0", "cooling",
         "bin_center", "bin_lo", "bin_hi", "count", "tol"],
    )
    counts["optimization_results.json"] = len(bundle["cells"])

    baseline = [
        row for row in summary_rows
        if row["T0"] == BASE_T0 and row["cooling"] == BASE_COOLING
    ]
    baseline_comparisons = [
        r for r in bundle["comparisons"]
        if r["T0"] == BASE_T0 and r["cooling"] == BASE_COOLING
    ]
    hypo_summary = summarize_step_width_hypothesis(bundle)
    payload = {
        "meta": bundle["meta"],
        "cell_summary": summary_rows,
        "baseline_cells": baseline,
        "aggregates": aggregate_cells(bundle),
        "comparisons_baseline": baseline_comparisons,
        "comparisons_total_rows": len(bundle["comparisons"]),
        "hypothesis_summary": hypo_summary,
        "step_width_verdict": step_width_verdict(hypo_summary),
        "interpretation_notes": build_interpretation_notes(bundle),
        "rows_written": counts,
    }
    with (RESULTS_DIR / "optimization_results.json").open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, ensure_ascii=False, indent=2)
    return counts


# ============================================================
# 10. 可視化（ダークテーマ・図内テキストは英語）
# ============================================================

FIGSIZE = (19.20, 12.00)
FIG_DPI = 100
PANEL_BG = "#141414"


def _new_figure(suptitle: str, subtitle: str = "") -> "plt.Figure":
    fig = plt.figure(figsize=FIGSIZE, dpi=FIG_DPI, facecolor=DARK_BG)
    fig.suptitle(suptitle, color=DARK_TEXT, fontsize=20, fontweight="bold", y=0.965)
    if subtitle:
        fig.text(0.5, 0.928, subtitle, color=DARK_SUBTLE, fontsize=12, ha="center")
    return fig


def _style_ax(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_facecolor(DARK_BG)
    if title:
        ax.set_title(title, color=DARK_TEXT, fontsize=13, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=DARK_TEXT, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=DARK_TEXT, fontsize=11)
    ax.tick_params(colors=DARK_SUBTLE, labelsize=9)
    ax.grid(True, color=DARK_GRID, lw=0.6, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(DARK_GRID)


def _legend(ax, loc: str = "best", fontsize: float = 9) -> None:
    ax.legend(fontsize=fontsize, loc=loc, facecolor=PANEL_BG,
              edgecolor=DARK_GRID, labelcolor=DARK_TEXT, framealpha=0.9)


def _cell_lookup(bundle: dict) -> dict:
    """(objective, T0, cooling, noise) -> stats dict"""
    return {
        (c.objective_key, c.T0_value, c.cooling, c.noise): c.stats()
        for c in bundle["cells"]
    }


def bundle_noises(bundle: dict) -> list[str]:
    """バンドルに実際に含まれるノイズ条件（図・表の並び順に使う）。"""
    declared = bundle["meta"]["params"].get("noise_conditions") or list(NOISE_TYPES)
    present = {c.noise for c in bundle["cells"]}
    ordered = [n for n in declared if n in present]
    ordered += [n for n in present if n not in ordered]
    return ordered


def plot_escape_demo(bundle: dict, n_demo_trials: int = 12, demo_steps: int = 1500) -> Path:
    """
    代表軌跡のデモ（統計の根拠ではなく『動きの絵』）。

    スカラー版 simulated_annealing を条件ごとに独立ストリームで走らせ、
    πが取り残された試行を1つ選んで 1D Rastrigin 上で描く。
    """
    spec = get_objective("rastrigin_1d")
    sources = bundle["noise_sources"]
    conditions = ["pi", "inverse_pi", "inverse_pi_mean_matched"]
    base_entropy = int(bundle["meta"]["params"]["seed_entropy"])
    runs: dict[str, list[SATrialResult]] = {}

    for idx, cond in enumerate(conditions):
        results = []
        for t in range(n_demo_trials):
            res = simulated_annealing(
                sources[cond],
                T0=BASE_T0,
                cooling=BASE_COOLING,
                steps=demo_steps,
                step_scale=STEP_SCALE,
                threshold=spec.tol,
                objective=spec,
                seed_entropy=base_entropy + 5000 + idx * 100 + t,
            )
            results.append(res)
        runs[cond] = results

    # 対比が最も鮮やかな試行を選ぶ（π が未達かつ統制逆π が到達を優先）。
    # ただし「初期位置がたまたま tol 内」の試行は脱出の絵にならないので除外する。
    best_t, best_score = 0, -1e9
    for t in range(n_demo_trials):
        pi_d = runs["pi"][t].final_distance
        iv_d = runs["inverse_pi"][t].final_distance
        mm_d = runs["inverse_pi_mean_matched"][t].final_distance
        started_inside = any(
            runs[c][t].steps_to_threshold == 0 for c in conditions
        )
        score = (pi_d - mm_d) + 0.5 * (pi_d - iv_d)
        if pi_d >= spec.tol and not started_inside and score > best_score:
            best_score, best_t = score, t

    fig = _new_figure(
        "Trajectory Demo on 1D Rastrigin  (illustration, not evidence)",
        f"T0={BASE_T0:g}, cooling={BASE_COOLING:g}, step_scale={STEP_SCALE:g}, "
        f"demo trial #{best_t} of {n_demo_trials} (independent RNG stream per condition)",
    )
    gs = fig.add_gridspec(2, 1, hspace=0.32, height_ratios=[1.05, 1.0])

    ax_landscape = fig.add_subplot(gs[0])
    xs = np.linspace(-5.0, 5.0, 4000)
    ax_landscape.plot(xs, rastrigin_1d(xs), color="#555555", lw=1.0, label="Rastrigin 1D")
    for cond in conditions:
        traj = runs[cond][best_t].trajectory
        ax_landscape.plot(traj, rastrigin_1d(traj), color=NOISE_COLORS[cond],
                          lw=1.0, alpha=0.85, label=NOISE_LABELS[cond])
    ax_landscape.set_ylim(-1.0, 40.0)
    ax_landscape.axvline(0.0, color="white", ls="--", lw=1.0, alpha=0.5)
    ax_landscape.axvspan(-spec.tol, spec.tol, color="white", alpha=0.05)
    _style_ax(ax_landscape, xlabel="x", ylabel="f(x)", title="Where each noise ended up")
    _legend(ax_landscape, loc="upper right")

    ax_time = fig.add_subplot(gs[1])
    for cond in conditions:
        res = runs[cond][best_t]
        ax_time.plot(np.arange(len(res.trajectory)), res.trajectory,
                     color=NOISE_COLORS[cond], lw=1.1, alpha=0.9,
                     label=f"{NOISE_LABELS[cond]} (arrival={res.steps_to_threshold})")
    ax_time.axhspan(-spec.tol, spec.tol, color="white", alpha=0.06)
    ax_time.axhline(0.0, color="white", ls="--", lw=1.0, alpha=0.5)
    _style_ax(ax_time, xlabel="SA step", ylabel="x",
              title="x over time (same demo trial)")
    _legend(ax_time, loc="upper right")

    path = FIGURES_DIR / "optimization_escape.png"
    fig.savefig(path, dpi=FIG_DPI, facecolor=DARK_BG)
    plt.close(fig)
    return path


def plot_success_rate_heatmap(bundle: dict) -> Path:
    """成功率ヒートマップ: 行=目的関数, 列=ノイズ条件, パネル=T0（冷却率は基準値）。"""
    lookup = _cell_lookup(bundle)
    noises = bundle_noises(bundle)
    obj_keys = [s.key for s in OBJECTIVES]
    obj_labels = [s.label for s in OBJECTIVES]

    fig = _new_figure(
        "Success rate by noise condition x objective x temperature band",
        f"cooling={BASE_COOLING:g}, {MATRIX_TRIALS} trials/cell. "
        "Shade = fraction of trials ending within tol of the known optimum.",
    )
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.18, top=0.87, bottom=0.07,
                          left=0.11, right=0.88)

    im = None
    for panel, t0_value in enumerate(MATRIX_T0_LIST):
        ax = fig.add_subplot(gs[panel // 2, panel % 2])
        matrix = np.full((len(obj_keys), len(noises)), np.nan)
        for i, obj_key in enumerate(obj_keys):
            for j, noise in enumerate(noises):
                stat = lookup.get((obj_key, t0_value, BASE_COOLING, noise))
                if stat:
                    matrix[i, j] = stat["success_rate"]
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(noises)))
        ax.set_xticklabels([NOISE_LABELS[n].replace(" ", "\n") for n in noises],
                           fontsize=8, color=DARK_SUBTLE)
        ax.set_yticks(range(len(obj_labels)))
        ax.set_yticklabels(obj_labels, fontsize=9, color=DARK_SUBTLE)
        for i in range(len(obj_keys)):
            for j in range(len(noises)):
                val = matrix[i, j]
                if not math.isnan(val):
                    ax.text(j, i, f"{val * 100:.0f}", ha="center", va="center",
                            fontsize=9,
                            color="black" if val > 0.55 else DARK_TEXT)
        _style_ax(ax, title=f"T0 = {t0_value:g}")
        ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=DARK_SUBTLE)
    cbar.outline.set_edgecolor(DARK_GRID)
    plt.setp(cbar.ax.get_yticklabels(), color=DARK_SUBTLE, fontsize=9)
    cbar.set_label("success rate", color=DARK_TEXT, fontsize=11)

    path = FIGURES_DIR / "success_rate_comparison.png"
    fig.savefig(path, dpi=FIG_DPI, facecolor=DARK_BG)
    plt.close(fig)
    return path


def plot_step_distribution(bundle: dict) -> Path:
    """
    『歩幅そのもの』の比較（3パネル）。

    左: 提案ステップ幅 |step| の分布（受理に依らず決まる量）
    中: 平均歩幅。mean-matched が π 非ゼロ平均へ寄っていることを確認
    右: ゼロステップ率。π だけが約1割静止する
    ※ 平均歩幅とゼロ率はスケールが違うので同一軸に重ねない。
    """
    sources = bundle["noise_sources"]
    noises = bundle_noises(bundle)
    steps = int(bundle["meta"]["params"]["n_steps"])
    widths_map = {
        noise: step_width_sequence(sources[noise], steps, BASE_T0, BASE_COOLING)
        for noise in noises
    }
    nonzero_pi = float(np.mean([d for d in sources["pi"] if d != 0]))
    colors = [NOISE_COLORS[n] for n in noises]

    fig = _new_figure(
        "Step-width design: what the mean-matched condition actually controls",
        f"T0={BASE_T0:g}, cooling={BASE_COOLING:g}, step_scale={STEP_SCALE:g}, "
        f"pi non-zero digit mean = {nonzero_pi:.3f} (left axis of middle panel is step width only)",
    )
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.05, right=0.97, top=0.84, bottom=0.13)

    # --- 左: 歩幅ヒストグラム ---
    ax_hist = fig.add_subplot(gs[0, 0])
    hi = max(float(np.nanmax([w.max() for w in widths_map.values()])), 1e-9)
    bins = np.linspace(0.0, hi, 60)
    for noise in noises:
        positive = widths_map[noise]
        positive = positive[positive > 0.0]
        ax_hist.hist(positive, bins=bins[1:], histtype="step", lw=1.6,
                     color=NOISE_COLORS[noise], label=NOISE_LABELS[noise], alpha=0.9)
    ax_hist.set_yscale("log")
    _style_ax(ax_hist, xlabel="|proposed step|  (non-zero steps only)",
              ylabel="count of steps",
              title="Step size distribution")
    _legend(ax_hist, loc="upper right", fontsize=8)

    # --- 中: 平均歩幅 ---
    xpos = np.arange(len(noises))
    labels = [NOISE_LABELS[n].replace(" ", "\n") for n in noises]
    ax_w = fig.add_subplot(gs[0, 1])
    means = [float(widths_map[n].mean()) for n in noises]
    bars = ax_w.bar(xpos, means, width=0.68, color=colors, alpha=0.9)
    for rect, val in zip(bars, means):
        ax_w.text(rect.get_x() + rect.get_width() / 2.0, val, f"{val:.3f}",
                  ha="center", va="bottom", fontsize=9, color=DARK_TEXT)
    ax_w.set_xticks(xpos)
    ax_w.set_xticklabels(labels, fontsize=8, color=DARK_SUBTLE)
    ax_w.set_ylim(0.0, max(means) * 1.18)
    _style_ax(ax_w, ylabel="mean |step|",
              title="Mean step width (all steps)")

    # --- 右: ゼロステップ率 ---
    ax_z = fig.add_subplot(gs[0, 2])
    zeros = [float(np.mean(widths_map[n] == 0.0)) for n in noises]
    bars_z = ax_z.bar(xpos, zeros, width=0.68, color=colors, alpha=0.9)
    for rect, val in zip(bars_z, zeros):
        ax_z.text(rect.get_x() + rect.get_width() / 2.0, val, f"{val * 100:.1f}%",
                  ha="center", va="bottom", fontsize=9, color=DARK_TEXT)
    ax_z.set_xticks(xpos)
    ax_z.set_xticklabels(labels, fontsize=8, color=DARK_SUBTLE)
    ax_z.set_ylim(0.0, max(max(zeros) * 1.35, 0.02))
    _style_ax(ax_z, ylabel="zero-step rate",
              title="Frozen steps (digit = 0)")

    path = FIGURES_DIR / "step_distribution_comparison.png"
    fig.savefig(path, dpi=FIG_DPI, facecolor=DARK_BG)
    plt.close(fig)
    return path


def plot_convergence(bundle: dict) -> Path:
    """目的関数ごとの『最適解までどれだけ近づいたか』曲線（基準温度帯）。"""
    cells = {
        (c.objective_key, c.noise): c
        for c in bundle["cells"]
        if c.T0_value == BASE_T0 and c.cooling == BASE_COOLING
    }
    noises = bundle_noises(bundle)

    fig = _new_figure(
        "Convergence: mean distance to the known optimum (solid) and best-so-far (dashed)",
        f"T0={BASE_T0:g}, cooling={BASE_COOLING:g}. Solid lines match the success criterion "
        "(where the sampler actually ends up); dashed best-so-far saturates for all conditions.",
    )
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.20, top=0.86, bottom=0.08,
                          left=0.06, right=0.98)

    for idx, spec in enumerate(OBJECTIVES):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        for noise in noises:
            cell = cells.get((spec.key, noise))
            if cell is None:
                continue
            ax.plot(cell.curve_steps, cell.curve_current, color=NOISE_COLORS[noise],
                    lw=1.7, alpha=0.95, label=NOISE_LABELS[noise])
            ax.plot(cell.curve_steps, cell.curve_best, color=NOISE_COLORS[noise],
                    lw=0.9, alpha=0.35, ls="--")
        ax.axhline(spec.tol, color="white", ls=":", lw=1.0, alpha=0.6)
        _style_ax(ax, xlabel="SA step", ylabel="mean distance to optimum",
                  title=f"{spec.label}  (tol={spec.tol:g})")
        if idx == 0:
            _legend(ax, loc="upper right", fontsize=8)

    path = FIGURES_DIR / "convergence_comparison.png"
    fig.savefig(path, dpi=FIG_DPI, facecolor=DARK_BG)
    plt.close(fig)
    return path


def plot_escape_time(bundle: dict) -> Path:
    """
    脱出までのステップ数と、温度帯による変化。

    左: 目的関数ごとの到達ステップ中央値（到達した試行のみ）
    中央・右: rastrigin_1d / rosenbrock の成功率 vs T0
    """
    lookup = _cell_lookup(bundle)
    noises = bundle_noises(bundle)
    obj_keys = [s.key for s in OBJECTIVES]

    fig = _new_figure(
        "Escape time and temperature dependence",
        "Left: median steps to first arrival (arrived trials only). "
        "Right panels: success rate across T0 bands.",
    )
    gs = fig.add_gridspec(1, 3, wspace=0.30, top=0.85, bottom=0.12,
                          left=0.05, right=0.98)

    # --- 左: 到達ステップ中央値 ---
    ax_left = fig.add_subplot(gs[0, 0])
    n_cond = len(noises)
    xbase = np.arange(len(obj_keys), dtype=float)
    width = 0.8 / n_cond
    for j, noise in enumerate(noises):
        vals, errs_lo, errs_hi = [], [], []
        for obj_key in obj_keys:
            stat = lookup.get((obj_key, BASE_T0, BASE_COOLING, noise))
            med = stat["escape_step_median"] if stat else float("nan")
            vals.append(0.0 if math.isnan(med) else med)
            errs_lo.append(0.0)
            errs_hi.append(0.0)
        pos = xbase + (j - (n_cond - 1) / 2.0) * width
        ax_left.bar(pos, vals, width=width * 0.92, color=NOISE_COLORS[noise],
                    alpha=0.88, label=NOISE_LABELS[noise])
    ax_left.set_xticks(xbase)
    ax_left.set_xticklabels([s.label.replace(" ", "\n") for s in OBJECTIVES],
                            fontsize=8, color=DARK_SUBTLE)
    _style_ax(ax_left, ylabel="median steps to arrival",
              title="Escape time (0 = never reached)")
    _legend(ax_left, loc="upper left", fontsize=7)

    # --- 中央 / 右: 成功率 vs T0 ---
    for panel, obj_key in enumerate(("rastrigin_1d", "rosenbrock")):
        ax = fig.add_subplot(gs[0, 1 + panel])
        for noise in noises:
            ys = []
            for t0_value in MATRIX_T0_LIST:
                stat = lookup.get((obj_key, t0_value, BASE_COOLING, noise))
                ys.append(stat["success_rate"] if stat else float("nan"))
            ax.plot(MATRIX_T0_LIST, ys, marker="o", ms=6, lw=1.8,
                    color=NOISE_COLORS[noise], label=NOISE_LABELS[noise])
        ax.set_xticks(list(MATRIX_T0_LIST))
        _style_ax(ax, xlabel="T0 (initial temperature)", ylabel="success rate",
                  title=f"{get_objective(obj_key).label}")
        _legend(ax, loc="best", fontsize=7)

    path = FIGURES_DIR / "escape_time_comparison.png"
    fig.savefig(path, dpi=FIG_DPI, facecolor=DARK_BG)
    plt.close(fig)
    return path


def generate_figures(bundle: dict) -> tuple[list[Path], list[str]]:
    """5枚の図を作る。個別の失敗は止まらないが報告する。"""
    ensure_output_dirs()
    makers = [
        ("optimization_escape.png", plot_escape_demo),
        ("success_rate_comparison.png", plot_success_rate_heatmap),
        ("step_distribution_comparison.png", plot_step_distribution),
        ("convergence_comparison.png", plot_convergence),
        ("escape_time_comparison.png", plot_escape_time),
    ]
    saved: list[Path] = []
    failures: list[str] = []
    for name, maker in makers:
        try:
            saved.append(maker(bundle))
            print(f"      [FIG] {name}")
        except Exception as exc:  # pragma: no cover - 描画依存
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"      [WARN] {name} の生成に失敗: {exc}", file=sys.stderr)
    return saved, failures


# ============================================================
# 11. コンソールレポート（断定せず「示唆」と「条件」を書く）
# ============================================================

ZENN_ESSAY_TEMPLATE = """
===============================================================
Zenn 考察用テンプレート: 逆πノイズと「平均ステップ幅」の切り分け

◆ 観測したこと（数値が語る範囲）
  ・πノイズでは digit=0 のステップが約 {pi_zero_rate:.1f}% 発生し、その間は完全に停止する。
  ・逆πノイズは 0 を含まないが、同時に平均歩幅も π より大きい（{inv_mean:.2f} vs {pi_mean:.2f}）。
  ・そこで歩幅だけを π に揃えた統制条件 (inverse_pi_mean_matched) を作った
    （スケール係数 {scale:.4f}）。

◆ 歩幅を揃えたあとに残った差
  ・pi vs inverse_pi        : 成功率差の平均 {raw_gap:+.4f}
  ・pi vs mean-matched      : 成功率差の平均 {matched_gap:+.4f}
  ・差のうち {explained_pct:.0f}% 程度は歩幅の違いで説明がつく見立て。

◆ 言えること（示唆どまり）
  {verdict}

◆ 条件ごとの振れ
  歩幅を揃えても差が残る／向きが反転するセルは、温度帯と目的関数によって変わります。
  「一様に効く／効かない」ではなく、上の {n_cells} セル分の分布で読んでください。

◆ 言えないこと（断定しない）
  ・「円の不完全性が AI を救う」系の一般化は、このデータからはできない。
  ・1セル {n_trials} 試行の探索実験であり、効果の大小より向きを見る段階。
  ・同じ π 桁列を全条件で共有しているため、桁列固有の偏りは相関して残る。
===============================================================
"""


def _fmt_rate(value: float) -> str:
    return "  n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value * 100:5.1f}%"


def _fmt_num(value: float, spec: str = "8.3f") -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return " " * (len(spec) - 1) + "n/a"
    return f"{value:{spec}}"


def print_comparison_table(
    bundle: dict,
    objective: str = "rastrigin_1d",
    metric: str = "success_rate",
) -> None:
    """基準温度帯での 4 ペア比較。p 値はBonferroni 前/後を並べて見せる。"""
    rows = [
        r for r in bundle["comparisons"]
        if r["objective"] == objective and r["T0"] == BASE_T0
        and r["cooling"] == BASE_COOLING and r["metric"] == metric
    ]
    if not rows:
        return
    print()
    print(f"  --- 比較（{objective} / T0={BASE_T0:g} / cooling={BASE_COOLING:g} / {metric}） ---")
    for r in rows:
        ci = (
            f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
            if not (math.isnan(r["ci_lo"]) or math.isnan(r["ci_hi"]))
            else "      n/a     "
        )
        orr = (
            "  n/a" if math.isnan(r["odds_ratio"]) else f"{r['odds_ratio']:.2f}"
        )
        print(
            f"    {r['cond_a']:>9} vs {r['cond_b']:<26}"
            f" diff={r['diff']:+7.3f} CI={ci} OR={orr:>5}"
            f" p={r['p_value']:.4f} p_bonf={r['p_bonferroni_global']:.3f}"
        )


def step_width_verdict(hypo: dict) -> str:
    """
    平均差の縮み具合から、『平均ステップ幅説』に対する方向性を文章化する。
    断定を避け、しきい値で段階的に表現強さを変える。
    """
    raw = hypo.get("pi_vs_inverse_pi", {}).get("mean_diff_a_minus_b", float("nan"))
    matched = hypo.get("pi_vs_inverse_pi_mean_matched", {}).get("mean_diff_a_minus_b", float("nan"))
    if not (math.isfinite(raw) and math.isfinite(matched)):
        return "データが足りず判定できません。"
    shrink = abs(raw) - abs(matched)
    if abs(raw) < 0.02:
        return (
            f"pi との成功率差は元々小さく（逆π {raw:+.4f}／統制後 {matched:+.4f}）、"
            "この実験系では『逆πが強い』という前提自体が成立しにくい。"
        )
    if shrink > 0.5 * abs(raw):
        return (
            f"歩幅を揃えると成功率差の平均は {raw:+.4f} → {matched:+.4f} と半分以上縮み、"
            "平均ステップ幅説を支持する方向の結果。"
        )
    if shrink > 0:
        return (
            f"歩幅を揃えると差は縮むが残り得る（{raw:+.4f} → {matched:+.4f}）。"
            "歩幅以外の要因も混ざっている可能性。"
        )
    return (
        f"歩幅を揃えても差は縮まない（{raw:+.4f} → {matched:+.4f}）。"
        "この設定では平均ステップ幅説は支持されない。"
    )


def print_honest_summary(bundle: dict) -> None:
    """数値から『言えること／言えないこと』を印刷する。断定形は使わない。"""
    hypo = summarize_step_width_hypothesis(bundle)
    meta = bundle["meta"]
    noise_meta = meta["noise_meta"]

    print()
    print("=" * 78)
    print("  何が見えたか（示唆であって結論ではない）")
    print("=" * 78)
    for note in build_interpretation_notes(bundle):
        print(f"  - {note}")

    raw = hypo.get("pi_vs_inverse_pi", {})
    matched = hypo.get("pi_vs_inverse_pi_mean_matched", {})
    explained = hypo.get("success_rate_gap_explained_ratio", float("nan"))
    essay = ZENN_ESSAY_TEMPLATE.format(
        pi_zero_rate=noise_meta.get("pi_zero_rate", float("nan")) * 100.0,
        inv_mean=noise_meta.get("inverse_pi_mean", float("nan")),
        pi_mean=noise_meta.get("pi_mean", float("nan")),
        scale=noise_meta.get("inverse_pi_mean_matched_scale", float("nan")),
        raw_gap=raw.get("mean_diff_a_minus_b", float("nan")),
        matched_gap=matched.get("mean_diff_a_minus_b", float("nan")),
        explained_pct=(explained * 100.0) if isinstance(explained, float) and math.isfinite(explained) else float("nan"),
        n_trials=meta["params"]["n_trials_per_cell"],
        n_cells=raw.get("n_cells", 0),
        verdict=step_width_verdict(hypo),
    )
    print(essay)


def run_legacy_single_condition_comparison(
    n_trials: int = N_TRIALS,
    steps: int = N_STEPS,
    seed_entropy: int = SEED,
) -> dict:
    """
    旧版相当の単発比較（1D Rastrigin・π vs 逆π）。

    旧版はおなじシードを条件ごとに再設定していたが、それは擬似ペア化になるため、
    ここでも条件別に独立ストリームを導出する。出力は results/legacy_single_run.csv。
    """
    ensure_output_dirs()
    pi_digits = compute_pi_digits(NUM_DIGITS)
    sources, _ = build_noise_sources(pi_digits)
    spec = get_objective("rastrigin_1d")

    rows: list[dict] = []
    for idx, cond in enumerate(("pi", "inverse_pi")):
        for t in range(n_trials):
            res = simulated_annealing(
                sources[cond],
                T0=T0,
                cooling=COOLING,
                steps=steps,
                step_scale=STEP_SCALE,
                threshold=THRESHOLD,
                objective=spec,
                seed_entropy=seed_entropy + 7000 + idx * 1000 + t,
            )
            rows.append({
                "noise": cond,
                "trial": t,
                "final_x": res.final_x,
                "final_distance": res.final_distance,
                "success": int(res.steps_to_threshold is not None),
                "steps_to_threshold": (-1 if res.steps_to_threshold is None else res.steps_to_threshold),
                "accept_rate": res.accept_rate,
                "zero_steps": res.zero_steps,
                "mean_step_width": res.mean_step_width,
                "max_step_width": res.max_step_width,
            })
    _write_csv(
        RESULTS_DIR / "legacy_single_run.csv", rows,
        ["noise", "trial", "final_x", "final_distance", "success",
         "steps_to_threshold", "accept_rate", "zero_steps",
         "mean_step_width", "max_step_width"],
    )
    summary = {}
    for cond in ("pi", "inverse_pi"):
        sub = [r for r in rows if r["noise"] == cond]
        summary[cond] = {
            "success_rate": float(np.mean([r["success"] for r in sub])),
            "mean_abs_error": float(np.mean([r["final_distance"] for r in sub])),
            "median_abs_error": float(np.median([r["final_distance"] for r in sub])),
            "accept_rate": float(np.mean([r["accept_rate"] for r in sub])),
            "zero_step_rate": float(np.mean([r["zero_steps"] for r in sub]) / steps),
        }
    return summary


def print_baseline_table(bundle: dict, max_objectives: int = 6) -> None:
    """基準温度帯（T0=25, cooling=0.997）のセル一覧。まずここを読む。"""
    lookup = _cell_lookup(bundle)
    noises = bundle_noises(bundle)
    print()
    print("=" * 78)
    trials_per_cell = bundle["meta"]["params"]["n_trials_per_cell"]
    print(f"  基準温度帯セル: T0={BASE_T0:g}, cooling={BASE_COOLING:g}"
          f"（Wilson 95%CI 付き・{trials_per_cell}試行/セル）")
    print("=" * 78)
    for spec in OBJECTIVES[:max_objectives]:
        print(f"\n  [{spec.label}]  tol={spec.tol:g}  {spec.shape_note}")
        print(f"    {'condition':<28}{'success':>9}{'95%CI':>18}"
              f"{'arrived':>9}{'MAE':>9}{'escape_med':>12}{'zero%':>8}")
        for noise in noises:
            stat = lookup.get((spec.key, BASE_T0, BASE_COOLING, noise))
            if stat is None:
                continue
            ci_lo, ci_hi = wilson_interval(stat["n_success"], stat["n_trials"])
            ci = f"[{ci_lo * 100:.0f},{ci_hi * 100:.0f}]"
            print(
                f"    {NOISE_LABELS[noise]:<28}"
                f"{_fmt_rate(stat['success_rate']):>9}"
                f"{ci:>18}"
                f"{_fmt_rate(stat['arrival_rate']):>9}"
                f"{_fmt_num(stat['mae_mean'], '9.3f'):>9}"
                f"{_fmt_num(stat['escape_step_median'], '12.0f'):>12}"
                f"{stat['zero_step_rate'] * 100:>7.1f}%"
            )


# ============================================================
# 12. メイン実行
# ============================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="逆πノイズ『平均ステップ幅仮説』マトリクス実験",
    )
    parser.add_argument("--trials", type=int, default=MATRIX_TRIALS,
                        help="1セルあたりの試行回数（既定 %(default)s）")
    parser.add_argument("--steps", type=int, default=N_STEPS,
                        help="1試行の SA ステップ数（既定 %(default)s）")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="全実験の entropy 基準（既定 %(default)s）")
    parser.add_argument("--quick", action="store_true",
                        help="12試行×1000ステップで動作確認（開発用）")
    parser.add_argument("--no-figures", action="store_true",
                        help="PNG 生成をスキップ")
    parser.add_argument("--with-width-matched", action="store_true",
                        help="第2統制条件 inverse_pi_width_matched を追加実行（計6条件）")
    parser.add_argument("--legacy", action="store_true",
                        help="旧版相当の単発比較（1D Rastrigin・π vs 逆π）のみ実行")
    args = parser.parse_args(argv)

    n_trials = 12 if args.quick else args.trials
    steps = 1000 if args.quick else args.steps
    noises = active_noise_conditions(args.with_width_matched)
    n_cells = (len(noises) * len(OBJECTIVES)
               * len(MATRIX_T0_LIST) * len(MATRIX_COOLING_LIST))

    print("=" * 78)
    print("  anti_circle_optimizer.py — 逆πノイズ マトリクス実験")
    print(f"  条件: {len(noises)} noise × {len(OBJECTIVES)} objectives × "
          f"{len(MATRIX_T0_LIST)} T0 × {len(MATRIX_COOLING_LIST)} cooling = {n_cells} cells")
    print(f"  試行/セル: {n_trials}  ステップ/試行: {steps}  entropy: {args.seed}")
    print("=" * 78)

    if args.legacy:
        summary = run_legacy_single_condition_comparison(
            n_trials=min(n_trials, N_TRIALS), steps=steps, seed_entropy=args.seed
        )
        print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
        print("  ※ 旧版互換の単発比較のみ。マトリクスは --legacy なしで実行されます。")
        return 0

    print("\n[1/4] マトリクス実行中...")
    bundle = run_matrix_experiment(
        n_trials=n_trials,
        steps=steps,
        seed_entropy=args.seed,
        include_extra_noise=args.with_width_matched,
    )
    print(f"      完了: {bundle['meta']['n_cells']} cells / "
          f"{bundle['meta']['n_trial_rows']} trial rows / "
          f"{bundle['meta']['elapsed_seconds']:.1f}s")

    print("\n[2/4] results/ に書き出し中...")
    counts = write_outputs(bundle)
    for name, count in counts.items():
        print(f"      results/{name}  ({count} rows)")

    print("\n[3/4] 比較検定（事前登録 4ペア × 3指標・Bonferroni 補正）")
    print(f"      comparisons 行数: {len(bundle['comparisons'])}")
    print_baseline_table(bundle)
    print_comparison_table(bundle, "rastrigin_1d", "success_rate")
    print_comparison_table(bundle, "rastrigin_1d", "mae")
    print_comparison_table(bundle, "rosenbrock", "success_rate")

    if args.no_figures:
        print("\n[4/4] 図の生成はスキップ (--no-figures)")
    else:
        print("\n[4/4] figures/ に図を生成中...")
        saved, failures = generate_figures(bundle)
        for path in saved:
            print(f"      figures/{path.name}")
        for failure in failures:
            print(f"      [WARN] {failure}", file=sys.stderr)

    print_honest_summary(bundle)
    print("✅ 完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
















