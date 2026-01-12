# 2元LJポテンシャルによる液体Gaの肩構造再現

## 研究概要

液体ガリウム（Ga）の構造因子S(Q)に現れる**肩構造（shoulder）**を、2元Lennard-Jones（LJ）ポテンシャルモデルで再現することを目的とした研究。

## 背景

液体Gaは、S(Q)のQ = 2.8-3.5 Å⁻¹付近に特徴的な「肩」を持つ。これは単一のLJポテンシャルでは再現できず、Gaの局所構造に2種類の環境が共存していることを示唆している。

---

## 手法

### 2元モデルの設計

Gaを2つの仮想的な原子タイプ（Ga1, Ga2）に分類：

| タイプ | σ（LJ径） | 役割 |
|--------|----------|------|
| Ga1 | 2.97 Å (1.1×基準) | コンパクトな局所構造 |
| Ga2 | 2.43 Å (0.9×基準) | 開かれた局所構造 |

- 基準σ = 2.70 Å
- ε = 0.430 kcal/mol（全ペア共通）
- Ga1-Ga2相互作用のσ₁₂は独立パラメータとして最適化

### グリッドサーチ

以下のパラメータ空間を探索（35シミュレーション）：

- **σ₁₂比**: 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18
- **Ga1割合**: 45%, 48%, 50%, 52%, 55%

### シミュレーション条件

- 温度: 423.15 K (150°C)
- 原子数: 1000
- 数密度: 0.0522 atoms/Å³
- 平衡化: 50,000ステップ
- 本番計測: 100,000ステップ
- LAMMPS + GPU (Kokkos/CUDA)

### 評価指標

1. **R-factor**: S(Q)全体の一致度
2. **RMSE**: Q = 1.5-5.0 Å⁻¹でのRMSE
3. **RMSE_shoulder**: Q = 2.8-3.5 Å⁻¹（肩領域）でのRMSE

---

## 結果

### 最適パラメータ

| パラメータ | 値 |
|-----------|-----|
| σ₁₂ | 3.159 Å (1.17×基準) |
| Ga1割合 | 45% |
| Ga2割合 | 55% |
| **R-factor** | **0.0580** |
| RMSE | 0.0715 |
| **RMSE_shoulder** | **0.0259** |

### 上位5結果

| σ₁₂比 | Ga1% | R-factor | RMSE_shoulder |
|-------|------|---------|---------------|
| **1.17** | **45%** | **0.0580** | **0.0259** |
| 1.12 | 45% | 0.0592 | 0.0799 |
| 1.13 | 50% | 0.0598 | 0.0824 |
| 1.13 | 48% | 0.0600 | 0.0851 |
| 1.17 | 48% | 0.0604 | 0.0751 |

### 傾向

- **σ₁₂ = 1.15-1.17x**が最適ゾーン
- **Ga1 < 50%**（Ga2がやや優勢）でR-factorが改善
- 肩領域の再現には**σ₁₂ = 1.17x**が特に効果的

---

## Voronoi解析

最適パラメータで追加シミュレーションを実施し、局所構造を解析：

- **平均配位数**: 15.10
- **原子体積**: 13.41 Å³（双峰分布）
- **主要多面体**: `<0,0,0,3,6,4>` (3.5%), `<0,0,0,2,8,4>` (2.8%)

Ga1（小σ）とGa2（大σ）で異なる局所環境が確認され、これが肩構造の起源であることを支持。

---

## 載せるべき画像

### 必須（メイン結果）

1. **best_fit_overlay.png**
   - 最適パラメータでのS(Q)比較
   - 実験データ（黒点）とシミュレーション（青線）のオーバーレイ
   - 肩領域（オレンジ）をハイライト
   - `outputs/fine_search_shoulder/analysis/best_fit_overlay.png`

2. **rfactor_heatmap.png**
   - パラメータ空間でのR-factorヒートマップ
   - 最適点を赤星で表示
   - `outputs/fine_search_shoulder/analysis/rfactor_heatmap.png`

### 推奨（パラメータ依存性）

3. **gallery_all_sq.png**
   - 35個全シミュレーションの結果一覧
   - パラメータ変化による系統的変化を可視化
   - `outputs/fine_search_shoulder/analysis/gallery_all_sq.png`

4. **rmse_heatmap.png**
   - RMSEベースのヒートマップ（R-factorと比較用）
   - `outputs/fine_search_shoulder/analysis/rmse_heatmap.png`

### Voronoi解析

5. **voronoi_summary.png**
   - 配位数、体積、Voronoi面数の統計
   - `outputs/fine_search_shoulder/voronoi/voronoi_summary.png`

6. **voronoi_local_structure.png**
   - 五角形vs六角形の分布と肩構造の起源解釈
   - `outputs/fine_search_shoulder/voronoi/voronoi_local_structure.png`

7. **voronoi_type_comparison.png**
   - Ga1（Type 1）とGa2（Type 2）の局所構造比較
   - `outputs/fine_search_shoulder/voronoi/voronoi_type_comparison.png`

### 補足（必要に応じて）

8. **voronoi_coordination.png** - 配位数分布の詳細
9. **voronoi_polyhedra.png** - 多面体種の頻度分析
10. **voronoi_volume.png** - 体積分布（双峰性の確認）

---

## 結論

- 2元LJモデルで液体Gaの肩構造を**R-factor = 0.058**で再現
- 最適パラメータ: σ₁₂ = 1.17×基準, Ga1割合 = 45%
- 肩領域のRMSE = 0.026と極めて良好な一致
- Voronoi解析により、2つの局所環境の共存を確認

---

## ファイル構成

```
outputs/fine_search_shoulder/
├── analysis/
│   ├── metrics_summary.csv      # 数値結果
│   ├── best_fit_overlay.png     # 最適フィット図
│   ├── rfactor_heatmap.png      # R-factorマップ
│   ├── rmse_heatmap.png         # RMSEマップ
│   ├── gallery_all_sq.png       # 全結果一覧
│   └── sq_comparison_*.png      # 個別比較（35枚）
└── voronoi/
    ├── voronoi_summary.png
    ├── voronoi_local_structure.png
    ├── voronoi_type_comparison.png
    └── ...
```

---

## 使用スクリプト

- `scripts/run_fine_search_shoulder.py` - グリッドサーチ実行
- `scripts/analyze_fine_search_shoulder.py` - 結果解析・画像生成
- `scripts/run_voronoi_analysis.py` - Voronoi計算
- `scripts/analyze_voronoi.py` - Voronoi図作成
