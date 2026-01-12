# 03_bimodal_grid_search

二種類Ga原子モデルによるS(Q)ショルダー構造の再現（論文の主要結果）

## 実験概要

- **目的**: 異なる原子サイズを持つ2種類のGa粒子（Ga1, Ga2）を用いて、液体GaのS(Q)ショルダー構造を再現する最適パラメータを探索
- **手法**: LAMMPSによるMDシミュレーション + グリッドサーチ

## パラメータ設定

### 固定パラメータ
- 温度: 423.15 K (150°C)
- 基準σ: 2.70 Å
- ε: 0.430 kcal/mol（全相互作用で共通）
- Ga1-Ga1 σ: 2.97 Å (基準の1.1倍)
- Ga2-Ga2 σ: 2.43 Å (基準の0.9倍)

### 探索パラメータ
| パラメータ | 範囲 | 刻み | 段階数 |
|-----------|------|------|--------|
| σ₁₂/σ₁₁ | 1.12 〜 1.18 | 0.01 | 7 |
| Ga1比率 | 45% 〜 55% | 2-3% | 5 |

合計: 35通り

## 結果

### 最適パラメータ
- **σ₁₂ = 1.17** (= 3.159 Å)
- **Ga1比率 = 45%**
- R-factor = 0.058
- RMSE = 0.071
- ショルダー部RMSE = 0.026

### 出力ファイル
```
outputs/
├── rdf_files/          # 各パラメータのRDFデータ
│   └── out_sig12_XXX_ga1_YY.rdf
└── analysis/
    ├── metrics_summary.csv      # 全パラメータの評価指標
    ├── rfactor_heatmap.png      # ★論文図4（右）
    ├── gallery_all_sq.png       # ★論文図5
    ├── best_fit_overlay.png     # ★論文図6
    └── sq_comparison_*.png      # 個別比較図
```

## 実行方法

```bash
# シミュレーション実行
python scripts/run_fine_search_shoulder.py

# 解析
python scripts/analyze_fine_search_shoulder.py
```

## 論文での使用箇所

- **図4**: パラメータ探索結果のヒートマップ (`rfactor_heatmap.png`)
- **図5**: 全パラメータのS(Q)ギャラリー (`gallery_all_sq.png`)
- **図6**: 最適パラメータでのS(Q)比較 (`best_fit_overlay.png`)
- **図7**: ショルダー部分の拡大図（別途作成）
