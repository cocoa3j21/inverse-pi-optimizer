# cosmic_toy — 閉じた玩具宇宙における重力のみの崩壊・時間の矢・循環

## 公開ミラーと開発正本

| 役割 | 場所 |
| :--- | :--- |
| **公開ミラー（本リポジトリ）** | [inverse-pi-optimizer/cosmic_toy/](.) — スクリプト・README・代表図・Zenn 草案（巨大 CSV/JSON は `.gitignore`） |
| **開発正本** | [ai-budget-meter / Labs/cosmic_toy](https://github.com/cocoa3j21/ai-budget-meter/tree/master/Labs/cosmic_toy) — フル実行ログ・`results_new/` 等 |

逆πマトリクス実験は [`../README.md`](../README.md)。Labs 全体索引は [ai-budget-meter / Labs](https://github.com/cocoa3j21/ai-budget-meter/tree/master/Labs)。

## 概要

2 次元 Plummer 軟化付きニュートン自己重力系（散逸・放射なし）を対象に、以下の問いについて **leapfrog (KDK) 時間可逆積分** による数値実験を行っています。

- **Q1**: ほぼ一様な系は重力だけで凝縮するか
- **Q2**: 凝縮状態は『ビッグバン直後』に似るか（エントロピー代理で見る）
- **Q3**: 時間の矢を膨張でなく収縮方向に置くと何が変わるか
- **Q4**: 循環するか、一方向に終わるか
- **Q5**: 速度反転 / ラベル反転で過去に戻るか

詳細は `cosmic_collapse_time_arrow.py` および出力結果（CSV/JSON/レポート）を参照してください。

## 実験パラメータ

| 記号 | 値 / 説明 |
|------|-----------|
| `dimension` | 2 |
| `interaction` | Plummer 軟化付きニュートン自己重力 |
| `integrator` | leapfrog (kick-drift-kick) — 時間可逆 |
| `boundary` | 孤立系（外部ポテンシャルなし）。閉箱の対照として `reflecting_wall` を持つ |
| `n_particles` | 200 |
| `n_steps` | 3000 |
| `dt` | 0.005 |
| `softening` | 0.05 |
| `R0` | 1.0（基準半径） |
| `G_scaled` | 0.005 |
| `sample_every` | 5 |
| `box_half_width` | 2.5 |
| `grid_cells` | 24 |
| `wall_radius` | 1.6 |
| `retrace_lags` | 100, 250, 500, 750 |
| `seeds` | 各条件 5 シード（合計 25 ラン） |
| `conditions` | `bound_collapse`, `expanding`, `marginal`, `reflecting_wall`, `time_reversed` |

## 条件の定義（初期条件）

| 条件 | 初期配置 | 初期速度 | 意図 |
|---|---|---|---|
| `bound_collapse` | 半径 `R0` の一様な円盤（N=200, m=1） | ゼロ（冷たい） | 重力のみの自由収縮。全エネルギー < 0 |
| `expanding` | 同上 | 外向き `1.15 * v_es` | 単調膨張の対照（一方向に開く） |
| `marginal` | 同上 | 外向き `0.86 * v_es` | 全エネルギー ≒ 0 の臨界膨張 |
| `reflecting_wall` | 同上 | ゼロ（冷たい） | 原点中心 `wall_radius` の弾性反射壁＝閉箱の対照 |
| `time_reversed` | `bound_collapse` を `n_steps/2` 進めた状態 | その速度を反転 | 巻き戻し（解釈実験の核）。系列の往復 |

`v_es = sqrt(2 * G * N * m / R0)` は円盤の概算脱出速度で、`0.86` 倍付近で全エネルギーが 0 に近くなる（臨界）。



## 主要な指標

- `R_gyr`: 重心半径（ハロー優位のサイズ指標）
- `R_half`: 質量 50% までの半径
- `R_core20`: 質量 20% までの半径（コア収縮メトリクス）
- `shape_entropy`: 自己スケール形状エントロピー
- `box_entropy`: 固定格子に対する粒子分布エントロピー
- `speed_entropy`: 速度空間エントロピー
- `macrostate_distance`: 初期マクロ状態への最接近距離（最初の 10% は除外）。マクロ状態ベクトルは **幾何・エントロピー成分のみ**（`R_gyr, R_half, R_core20, shape, box, speed`）で、エネルギーは保存量なので除外している
- `n_bounces`: 衝突回数
- `energy_drift_max`: 相対エネルギー漂移 `max|E(t)-E0|/(|KE0|+|PE0|)`

## 結果の要約（示唆のみ・復旧版スクリプトの実測）

以下は `results_new/` の実行値（N=200・3000 steps・5 シード×5 条件＝25 ラン、実行 152.1s、CSV 15215 行）。**以前の実行（`results/`）とは初期条件の細部が異なるため数値も異なる。**

- **Q1（重力だけで凝縮するか）**: `R_gyr` 最小/初期 = **0.250 ± 0.032**、コア（20%質量半径）最小/初期 = **0.173 ± 0.029**、`R_half` = 0.202 ± 0.030。凝縮はコア側に強く現れ、後段でハローは最大 **4.18 ± 1.84 倍**まで開く＝同一実行にコア収縮とハロー膨張（負の比熱）の両相が現れる。`expanding` / `marginal` ではコア収縮 = 1.000（ほぼ起きない）。
- **Q2（凝縮状態は『ビッグバン直後』に似るか）**: エントロピー代理の終始差は `box` = **−1.148 ± 0.174**、`shape` = **−0.613 ± 0.314**、`speed` = **+2.525 ± 0.062** nats。3 代理の向きが揃わず（box/shape 減少・speed 増加）、教科書的な『凝縮→エントロピー増』を明確には再現しなかった。よって『似る』も『似ていない』も**裁定できない**。`box` は粒子の箱外漏出（bound 最大 0.082）で見かけ上減る点に注意。
- **Q3（時間の矢を収縮方向に置く）**: bound の box エントロピー単調増加割合は実時間 0.472、`R_gyr` 膨張向きに並べ替えると 0.502、収縮向きだと 0.499。並べ替えは物理量を変えない（同じ multiset）ので、変わるのは『単調に見えるか』だけ。`expanding` では 0.267 → 収縮向き 0.733 と反転する。
- **Q4（循環するか一方向か）**: 初期マクロ状態への最接近距離は bound **2.445 ± 0.073**、reflecting_wall **2.402 ± 0.056**（バウンス 14.4 ± 4.2 回）、expanding 6.488。今回の定義（幾何・エントロピー成分）では**閉箱と孤立系に明確な差は出ない**。膨張・臨界は最終 26.7 / 15.4 倍まで一方向に開く。サイズの振動とエントロピーの不可逆性は別物として測っている。
- **Q5-A（速度反転で past に戻るか）**: なめらかな孤立系は戻り誤差 rms/R0 が最大 **1.40e-12**（lag=750）で丸め誤差レベル。反射壁ありでは最大 **2.04e-01**（同）まで拡大し、不連続な衝突が誤差を増幅する。
- **Q5-B（ラベル反転テスト）**: 系列を逆順に読んでも物理量の集合は**完全一致**（multiset identical = True、5 指標すべて）。変わるのは時間ラベルとの相関の符号だけ（`box_entropy` の Spearman −0.431 → +0.431）。よって『時空が逆方向に動く』とは**主張しない**。
- **反転実行（time_reversed）**: エントロピー代理の終始差 = +0.023 ± 0.105、初期マクロ状態への最接近距離 = **0.115 ± 0.032**。反転系は初期マクロ状態の近くまで戻る（＝系列を往復し、その後ふたたび収縮する）。これは『一様（＝低エントロピーと仮定した状態）から始めて人工的に巻き戻した』設計上の話にすぎない。
- **数値の信頼性**: エネルギー保存の相対漂移 max|E(t)−E0| / (|KE0|+|PE0|) = **0.0009**（bound のシード最大）。leapfrog(KDK) は可逆で、時間反転テストの前提と整合する。

**重要**: 本実験は 2 次元・軟化ニュートン・放射なし・有限時間の玩具です。得られるのは保存量・相関・再現誤差であり、宇宙論の結論・証明ではありません。「宇宙の正体が判明した」とは書きません。

## 出力ファイル

復旧版スクリプト（2026-11-25 再構成）は **既存の `results/` `figures/` を上書きせず**、新しいディレクトリへ書き出します。

| 種類 | パス | 備考 |
|---|---|---|
| CSV | `results_new/cosmic_collapse_time_arrow.csv` | `record_type` = series / run_summary / pooled / retrace / label_reversal |
| JSON | `results_new/cosmic_collapse_time_arrow.json` | `model` / `params` / `per_seed` / `pooled_by_condition` / `interpretation_notes` / `tone_policy` |
| PNG | `figures_new/cosmic_collapse_time_arrow.png` | 6 パネル（サイズ比・コア収縮・プール統計・エントロピー代理・マクロ状態距離・反転誤差） |
| スモーク | `results_new_quick/`, `figures_new_quick/` | `--quick` 用の別ディレクトリ |
| 旧結果 | `results/`, `figures/` | 以前の実行分。**上書きしない**（比較用に残す） |

- `cosmic_collapse_time_arrow.py` — 本文
- `README.md` — このファイル

## トーンポリシー

すべての記述は **示唆** として読んでください。

- 「証明された」「宇宙の正体が判明した」とは**書かない**。
- 数値と矛盾する断定は避け、実測値とその範囲／シード間変動を提示する。
- 結論がデータによっては裁定できない場合は明示する（「裁定できません」「指標の限界」など）。
- 実験系は極めて理想化された玩具であることを常に添える。

## 再実行方法

出力はスクリプト自身の場所（`__file__`）基準で決まるため、実行ディレクトリに依存しない。既定は `results_new/` と `figures_new/` で、既存の `results/` `figures/` には一切触れない。

```powershell
cd cosmic_toy

# フルラン（N=200 / 3000 steps / 5 seeds × 5 conditions、約 150 秒）
python cosmic_collapse_time_arrow.py

# スモークテスト（600 steps / 1 seed → results_new_quick・figures_new_quick）
python cosmic_collapse_time_arrow.py --quick

# 出力先を明示したい場合
python cosmic_collapse_time_arrow.py --outdir T:/_scratch_ipi/out --figdir T:/_scratch_ipi/fig
```

- 日本語出力は stdout/stderr を UTF-8 に再設定しているが、コンソール側が cp932 の場合は `PYTHONIOENCODING=utf-8` を推奨。
- 依存は `numpy` と `matplotlib`（図がない場合はスキップして続行）。

## 復旧メモ（2026-11-25）

`cosmic_collapse_time_arrow.py` の本文が一時行方不明になったため、以下から再構成した。

- 実行ログ `T:/_scratch_ipi/cos_full.out` / `cos_full2.out`（コンソール出力の全セクション）
- 既存出力 `results/*.csv`（77 列のヘッダ＝出力仕様）と `results/*.json`（`params` / `model` / `per_seed` / `pooled_by_condition` / `interpretation_notes` / `tone_policy` のキー構造）
- 本 README のパラメータ表

再構成版は numpy で加速度・エネルギー・半径・エントロピーをベクトル化しており、**数値は以前の実行と一致しない**（同一の乱数シード・同一物理仕様を目指すが、初期条件の細部とマクロ状態ベクトルの定義を整えたため）。以前の結果は `results/` `figures/` にそのまま残してあり、混在しない。

また、解釈ノート（`interpretation_notes`）はすべて**実測値から分岐する**ように書き直した。具体的には、

- Q2: エントロピー代理 3 種の増減を数値で判定し、「再現した／再現しなかった／向みが揃わない」を切り替える
- Q4: 閉箱と孤立系の最接近距離を比較して優劣を切り替え、差がなければ「明確な差はない」と書く
- Q5-A: 反射壁あり／なしの誤差比が 10 倍以上のときだけ「増幅した」と書く

「証明された」「宇宙の正体が判明した」に相当する表現は使っていない（`tone_policy` 内での否定形のみの言及）。

## 用語一致メモ（2026-11-25）

**目的:** 公開 Zenn・草案・JSON（`interpretation_notes` / `tone_policy`）の**語彙**を突合し、断定を示唆・玩具モデル・解釈実験に揃えた。**数値・図・CSV・JSON ファイル本体は変更していない。**

| 観点 | 公開 Zenn [2fc28f602b0a6c](https://zenn.dev/cocoa3j21/articles/2fc28f602b0a6c) | `ZENN_ARTICLE_DRAFT.md` | `results_new/…json` |
| :--- | :--- | :--- | :--- |
| **トーン** | 冒頭「言えること／言わないこと」、脚注で `tone_policy` 引用 | 同構造・[^tone] 一致 | `tone_policy` 一文（証明・判明を**書かない**＋解釈実験） |
| **各 Q** | 見出し直下 **示唆:** | 同様 | `interpretation_notes[0–10]`（機械向け長文・裁定不能の分岐あり） |
| **Q2** | 3 代理の向き不整合・裁定しにくい | 同様 | [1–3] で「似る／似ない」**裁定できない** |
| **Q3** | 解釈実験の核心・物理的断定ではない | 同様 | [4] と `interpretation_vs_physics` |
| **Q5-B** | 「主張しないこと: 時空が逆方向…」 | 同様 | `not_claimed` + [8] |
| **Q5-A 表現** | integrator 整合の示唆（公開取得 2026-11-25） | **可逆 integrator 上の巻き戻し**（断定回避）[^q5a] | [7] はログ向け「past に戻る」— **公開文の方が柔らかい** |
| **数値** | 表記は四捨五入（例 −1.15） | 草案と同一 | **正** = JSON/CSV フル精度（例 −1.1481） |
| **総括** | おわりに＝示唆の束・測り方のメモ | 同様 | `interpretation_notes[11]` 総括の限界 |

**Zenn 貼付ブロック:** 突合時は公開本文と草案の**差分小**のため、逆π のような節別貼付ファイルは**未作成**。再投稿・大改稿時は [`ZENN_ARTICLE_DRAFT.md`](ZENN_ARTICLE_DRAFT.md) をエディタへそのまま貼る。

**索引:** [`cosmic_article.md`](cosmic_article.md)（公開 URL・同期状態・再貼付方針）。

## Zenn 記事

- **公開済み（示唆）**: [【Python】閉じた玩具宇宙で、重力だけ・時間の矢・循環を「示唆」として眺める](https://zenn.dev/cocoa3j21/articles/2fc28f602b0a6c)
- 原稿・同期索引: [`ZENN_ARTICLE_DRAFT.md`](ZENN_ARTICLE_DRAFT.md) · [`cosmic_article.md`](cosmic_article.md)

## 今後の予定

本実験は**玩具モデル・解釈実験**の**示唆**です。宇宙論や物理法則の確定とは書きません。

| # | 項目 | 状態 |
| :--- | :--- | :--- |
| 2 | `rotation_minimal` ミラー反映 | **完了** — [inverse-pi-optimizer/tree/main/rotation_minimal](https://github.com/cocoa3j21/inverse-pi-optimizer/tree/main/rotation_minimal) |
| 3 | Zenn 記事（cosmic） | **公開済み** — [Zenn 記事](https://zenn.dev/cocoa3j21/articles/2fc28f602b0a6c)（原稿: [`ZENN_ARTICLE_DRAFT.md`](ZENN_ARTICLE_DRAFT.md)） |
| 4 | 関連（rotation_minimal） | **別プロジェクト** — Zenn **公開済** [c04f92631055ed](https://zenn.dev/cocoa3j21/articles/c04f92631055ed)（本記事 URL とは混同しない） |

導線: [逆πマトリクス（Zenn）](https://zenn.dev/cocoa3j21/articles/b72a764450e23a) · [GitHub @cocoa3j21](https://github.com/cocoa3j21) · [Buy Me a Coffee](https://www.buymeacoffee.com/cocoa3j21)

## 参考

- `cosmic_collapse_time_arrow.py` 本文内の `interpretation_notes` フィールドに詳細な Q1–Q5 のコメントを埋め込んでいます。
- JSON フィールド `tone_policy` にも方針を記載しています。
- エントロピー代理の限界（box leakage、粗視化の定義依存性）は `notes` セクションに記載しています。
