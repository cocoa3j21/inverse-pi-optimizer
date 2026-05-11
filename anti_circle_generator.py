#!/usr/bin/env python3
"""
anti_circle_generator.py — 逆円周率（Inverse π）のランダムウォーク可視化

哲学的背景:
「円周率（π）が無限に続くのは、現実世界に『完全な円（静止・安定）』が存在しない証拠である。
現実の物質は、円になろうとして永遠に『微細な補正（エラー修復）』を繰り返している。
ならば、πの各桁を反転させた『逆円周率（各桁の10の補数）』の数列こそが、
完全な円を作ろうとする『補正の動き（フラクタルな微振動）』であり、
宇宙のエネルギー（動力）の源なのではないか？」

出力: anti_circle_energy.png (1920x1080)
"""

import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="mpmath")


# ============================================================
# 1. π 桁の取得
# ============================================================

def compute_pi_digits(num_digits: int = 10000) -> list[int]:
    """
    mpmath を使って π を計算し、小数点以下 num_digits 桁のリストを返す。
    失敗した場合はハードコードされたフォールバック値を使用する。
    """
    try:
        import mpmath as mp
        mp.mp.dps = num_digits + 10  # 余裕を持って多めに計算
        pi_str = mp.nstr(mp.pi, num_digits + 2)  # "3." + digits
        # 小数点以下の桁を抽出
        digits_str = pi_str.split(".")[1][:num_digits]
        digits = [int(ch) for ch in digits_str]
        print(f"[INFO] π を {num_digits} 桁計算しました (mpmath)")
        return digits
    except Exception as e:
        print(f"[WARN] mpmath による π 計算に失敗: {e}", file=sys.stderr)
        print("[WARN] フォールバック: ハードコードされた π 桁 (1000桁) を使用します")
        return _fallback_pi_digits()


def _fallback_pi_digits() -> list[int]:
    """mpmath が使えない場合のフォールバック π 1000桁"""
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
    # 必要なら反復して 10000桁に拡張（簡易的）
    digits = [int(ch) for ch in pi_1000]
    while len(digits) < 10000:
        digits.extend(digits[: min(10000 - len(digits), len(digits))])
    return digits[:10000]


# ============================================================
# 2. 逆πの生成
# ============================================================

def compute_inverse_pi(pi_digits: list[int]) -> list[int]:
    """
    逆π: 各桁 d に対して inverse_d = 10 - d
    d == 0 の場合は inverse_d = 10 (0 は「補正なし=完全な円」なので存在しない)
    値域: 1〜10
    """
    inverse = []
    for d in pi_digits:
        inv = 10 - d if d != 0 else 10
        inverse.append(inv)
    return inverse


# ============================================================
# 3. ランダムウォーク（軌跡計算）
# ============================================================

def random_walk(
    digits: list[int],
    num_directions: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    各ステップ i:
      角度 θ = digit * (2π / num_directions)
      次の位置 = 現在位置 + (cos(θ), sin(θ))
    戻り値: (xs, ys)  NumPy 配列

    π の軌跡:  num_directions=10  → 角度 0°, 36°, 72°, ..., 324°
      角度0° = 東方向 = 「完全な静止」を象徴
    Inverse π の軌跡: num_directions=11 → 角度 32.7°, 65.5°, ..., 327.3°
      角度0°が存在せず、「完全な補正不要」の方向を排除
    """
    n = len(digits)
    xs = np.zeros(n + 1, dtype=np.float64)
    ys = np.zeros(n + 1, dtype=np.float64)

    angle_step = 2.0 * math.pi / num_directions

    for i in range(n):
        theta = digits[i] * angle_step
        xs[i + 1] = xs[i] + math.cos(theta)
        ys[i + 1] = ys[i] + math.sin(theta)

    return xs, ys


# ============================================================
# 4. 可視化
# ============================================================

def build_figure(
    pi_x: np.ndarray,
    pi_y: np.ndarray,
    inv_x: np.ndarray,
    inv_y: np.ndarray,
) -> plt.Figure:
    """π と逆π の軌跡を並べた 1920x1080 画像を生成"""
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(19.20, 10.80), dpi=100,
        facecolor="#0a0a0a",
    )

    # ---------- 左パネル: π ----------
    _draw_trajectory(
        ax=ax_left,
        xs=pi_x, ys=pi_y,
        title="π (Pi) — The Uncertainty",
        color="#00ffff",
        glow_color="#0088ff",
    )

    # ---------- 右パネル: 逆π ----------
    _draw_trajectory(
        ax=ax_right,
        xs=inv_x, ys=inv_y,
        title="Inverse π — The Correction Energy",
        color="#ff8800",
        glow_color="#ff4400",
    )

    # ---------- 最終距離テキスト ----------
    pi_dist = math.hypot(pi_x[-1], pi_y[-1])
    inv_dist = math.hypot(inv_x[-1], inv_y[-1])

    fig.text(
        0.25, 0.02,
        f"Final distance: {pi_dist:.2f}",
        color="white", fontsize=14,
        ha="center", va="bottom",
        family="sans-serif",
    )
    fig.text(
        0.75, 0.02,
        f"Final distance: {inv_dist:.2f}",
        color="white", fontsize=14,
        ha="center", va="bottom",
        family="sans-serif",
    )

    plt.tight_layout(pad=1.5)
    return fig


def _draw_trajectory(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    title: str,
    color: str,
    glow_color: str,
) -> None:
    """1 つのパネルに軌跡を描画"""
    ax.set_facecolor("#0a0a0a")
    ax.set_title(title, color="white", fontsize=18, fontweight="bold", pad=12)

    # 軸を非表示（微かなグリッドのみ）
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(color="#333333")

    # メイン軌跡
    ax.plot(xs, ys, lw=0.5, alpha=0.6, color=color)

    # グロー効果: 太く半透明で重ね描き
    ax.plot(xs, ys, lw=4, alpha=0.08, color=glow_color)
    ax.plot(xs, ys, lw=8, alpha=0.04, color=glow_color)

    # スキャッターポイントによる発光粒子
    step = max(1, len(xs) // 200)  # 約200点
    ax.scatter(
        xs[::step], ys[::step],
        s=1, alpha=0.15, color=color,
        zorder=2,
    )

    # スタート地点（白）
    ax.scatter(
        xs[0], ys[0],
        c="white", s=30, alpha=0.8, zorder=5,
        edgecolors="white", linewidth=0.5,
    )
    # エンド地点（軌跡の色+白枠）
    ax.scatter(
        xs[-1], ys[-1],
        c=color, s=50, alpha=0.9, zorder=5,
        edgecolors="white", linewidth=0.5,
    )

    # アスペクト比を等しく
    ax.set_aspect("equal")


# ============================================================
# 5. Zenn 考察 (実行時にも出力)
# ============================================================

ZENN_ESSAY = """
================================================================
Zenn考察: 「逆円周率（Inverse π）—— 不完全性が生む動力」

◆ π（円周率）のランダムウォーク
  → 10等分された角度（0°, 36°, 72°, …, 324°）の上を歩く。
  0°（東方向）も対等に存在するため、全方位への移動が均等。
  結果として軌跡は中心付近に留まり、大きなドリフトは生まれない。
  ── これは「完全な円（安定・静止）」の幻想を彷徨っているに過ぎない。

◆ 逆π（補正エネルギー）のランダムウォーク
  → 各桁を 10−d で反転させた数列（値域1〜10）を、11等分された角度
  （約32.7°, 65.5°, …, 327.3°）の上で歩かせる。
  値域に「0」が存在しないため、角度0°（完全な静止＝補正不要の方向）
  は永遠に訪れない。
  非対称な角度分布により、特定の方向へ不可避のドリフトが生じ、
  結果としてπよりもはるかに大きな移動距離（エネルギー）を獲得する。

◆ 結論
  「完全を目指す不完全さ」── 角度0°の欠如という微細な非対称性が、
  宇宙の動力（エネルギー）を生み出しているのではないか。
  逆πは「完全な円になろうとする諦めない補正の動き」そのものの可視化である。

================================================================
"""


# ============================================================
# 6. メイン実行
# ============================================================

def main():
    print("=" * 60)
    print("  anti_circle_generator.py — Inverse π Random Walk")
    print("=" * 60)

    NUM_DIGITS = 10000

    # Step 1: π 桁を取得
    print(f"\n[1/4] π を {NUM_DIGITS} 桁計算中...")
    pi_digits = compute_pi_digits(NUM_DIGITS)
    print(f"      取得完了: {len(pi_digits)} 桁")

    # Step 2: 逆π を生成
    print("\n[2/4] 逆π（補数数列）を生成中...")
    inv_digits = compute_inverse_pi(pi_digits)
    print(f"      生成完了: {len(inv_digits)} 桁 (値域 {min(inv_digits)}–{max(inv_digits)})")
    # 検証: 逆πに 0 が含まれていないこと
    assert 0 not in inv_digits, "逆πに0が含まれています！"
    print("      検証OK: 逆πに 0（完全な静止方向）は存在しません")

    # Step 3: ランダムウォーク計算
    # π は 10 等分 (36°刻み)、逆π は 11 等分で角度0°を排除
    # これにより逆πの軌跡は「完全な静止方向」のない非対称な動きを得る
    print("\n[3/4] ランダムウォーク軌跡を計算中...")
    pi_x, pi_y = random_walk(pi_digits, num_directions=10)
    inv_x, inv_y = random_walk(inv_digits, num_directions=11)
    pi_dist = math.hypot(pi_x[-1], pi_y[-1])
    inv_dist = math.hypot(inv_x[-1], inv_y[-1])
    print(f"      π の最終距離:     {pi_dist:.4f}")
    print(f"      逆π の最終距離:    {inv_dist:.4f}")
    print(f"      距離比 (逆π/π):    {inv_dist / pi_dist:.4f}")

    # Step 4: 可視化 & 保存
    print("\n[4/4] 画像を生成中...")
    fig = build_figure(pi_x, pi_y, inv_x, inv_y)
    output_path = Path(__file__).resolve().parent / "anti_circle_energy.png"
    fig.savefig(
        str(output_path),
        dpi=100,
        facecolor="#0a0a0a",
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.close(fig)
    print(f"      保存完了: {output_path}")

    # Zenn 考察を表示
    print(ZENN_ESSAY)

    print("\n✅ 完了！")


if __name__ == "__main__":
    main()
