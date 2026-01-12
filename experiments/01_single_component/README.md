# 01_single_component

単一成分LJモデルによる液体Gaのシミュレーション

## 目的

通常のLJポテンシャルでは液体GaのS(Q)ショルダー構造が再現されないことを確認する（論文4.1, 4.2節）

## パラメータ

| パラメータ | 値 |
|-----------|-----|
| σ | 2.70 Å |
| ε | 0.430 kcal/mol (= 1.8 kJ/mol) |
| 温度 | 423.15 K (150°C) |
| 原子数 | 1000 |
| カットオフ | 12.0 Å |

## 実行方法

### 1. シミュレーション実行（GPUインスタンス上で）

```bash
cd /home/yuki/lammps_settings_obata/experiments/01_single_component
python scripts/run_single_component_423K.py
```

### 2. 解析・図作成

```bash
python scripts/analyze_and_plot.py
```

## 出力ファイル

```
outputs/423K/
├── in.ga_single_423K           # LAMMPS入力ファイル
├── log.ga_single_423K          # LAMMPSログ
├── ga_single_423K.rdf          # g(r)データ
├── ga_single_423K_final.data   # 最終構造
├── sq_423K.dat                 # S(Q)データ
├── rdf_comparison_423K.png     # g(r)比較図
├── sq_comparison_423K.png      # S(Q)比較図
└── metrics.txt                 # 評価指標
```

## 論文での使用

- **図4.1**: `rdf_comparison_423K.png` → `thesis/figures/rdf_comparison_with_exp.png`
- **図4.2**: `sq_comparison_423K.png` → `thesis/figures/sq_comparison_with_exp.png`

## 期待される結果

- S(Q)の第一ピークは対称的（ショルダーなし）
- 実験データとの比較でショルダー領域（Q = 2.8–3.5 Å⁻¹）に不一致
