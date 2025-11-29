# Dissolve セットアップガイド - EGaInシステムのEPSR解析

## 概要

DissolveはEPSR（Empirical Potential Structure Refinement）の次世代版として開発された、中性子散乱データ解析用のオープンソースソフトウェアです。

### Dissolveの特徴

- ✅ EPSRアルゴリズムの正式な実装（A.K. Soper研究グループ開発）
- ✅ 完全なクラシカル力場サポート
- ✅ 100万原子規模のシミュレーション対応
- ✅ GUI + コマンドライン両対応
- ✅ 活発に開発中（GPL-3.0ライセンス）

---

## 1. インストール方法

### オプションA: プリビルドパッケージ（推奨）

**公式ダウンロードページ:**
```
https://projectdissolve.com/packages
```

**対応プラットフォーム:**
- Windows
- macOS
- Linux（複数ディストリビューション）

### オプションB: ソースからビルド

```bash
# リポジトリをクローン
git clone https://github.com/disorderedmaterials/dissolve.git
cd dissolve

# ビルド手順は以下を参照
# https://docs.projectdissolve.com/
```

---

## 2. 公式ドキュメント・リソース

| リソース | URL |
|---------|-----|
| 公式サイト | https://projectdissolve.com/ |
| ドキュメント | https://docs.projectdissolve.com/ |
| GitHub | https://github.com/disorderedmaterials/dissolve |
| チュートリアル | https://docs.projectdissolve.com/examples/ |
| 論文 | https://www.tandfonline.com/doi/abs/10.1080/00268976.2019.1651918 |

---

## 3. 推奨チュートリアル

### 初心者向け

1. **Argonチュートリアル** - 基本的なワークフロー
   - URL: https://docs.projectdissolve.com/examples/argon/
   - 内容: 液体アルゴンのシミュレーション、実験データとの比較
   - 難易度: ⭐ (初級)

2. **Liquid Water** - 液体構造の解析
   - URL: https://docs.projectdissolve.com/examples/water/
   - 内容: 298Kの液体水の構造解析
   - 難易度: ⭐⭐ (中級)

3. **Post Processing** - 外部データの使用
   - URL: https://docs.projectdissolve.com/examples/post-processing/
   - 内容: 外部シミュレーションデータ（LAMMPS等）の処理
   - 難易度: ⭐⭐⭐ (上級)

---

## 4. EGaInシステムへの応用

### 4.1 現在のデータを使用する方法

Dissolveは以下のワークフローで既存データを活用できます:

```
実験データ (data/g_exp_cleaned.dat)
         ↓
    Dissolve インポート
         ↓
  LAMMPS初期構造 (オプション)
         ↓
   Dissolveで EPSR実行
         ↓
    最適化された構造
```

### 4.2 セットアップ手順（概要）

#### ステップ1: プロジェクト作成

```bash
# Dissolve GUIを起動
dissolve-gui

# または コマンドライン
dissolve -c myproject.txt
```

#### ステップ2: スピーシーズ定義

EGaInシステムの場合:
- **Ga原子**: 質量 69.723、中性子散乱長 7.288 fm
- **In原子**: 質量 114.818、中性子散乱長 4.061 fm
- **組成**: Ga₀.₈₅₈In₀.₁₄₂

#### ステップ3: 力場設定

現在のLAMMPSパラメータを使用可能:
- Lennard-Jones パラメータ
- EAM ポテンシャル（使用中の場合）

#### ステップ4: 実験データのインポート

```
File → Import → Neutron Data
→ data/g_exp_cleaned.dat を選択
```

**データ形式**: 2カラム形式（r, g(r)）
```
# r(Å)    g(r)
2.0      0.0
2.1      0.05
...
```

#### ステップ5: EPSR実行

DissolveのEPSRモジュールを有効化し、自動的に以下を実行:
1. Reference Potential (RP) でMCシミュレーション
2. g(r), S(Q) の計算
3. Empirical Potential (EP) の計算
4. RP + EP で再シミュレーション
5. 収束まで繰り返し

---

## 5. LAMMPSとの統合

### オプション1: Dissolve単独使用（推奨）

Dissolve内蔵のMCエンジンを使用し、完全にDissolve内で解析。

**メリット**:
- EPSRアルゴリズムが正確に実装済み
- 設定が簡単
- 信頼性が高い

### オプション2: LAMMPSで初期構造生成 + Dissolveで解析

```bash
# 1. LAMMPSでシステムを平衡化
lmp -in in.egain_epsr_H100 -log equilibration.log

# 2. 構造をエクスポート
# LAMMPS input に追加:
# write_data equilibrated.data

# 3. Dissolveにインポート
# File → Import → LAMMPS Data File
```

### オプション3: Post-Processing モード

LAMMPSトラジェクトリをDissolveで解析:

```bash
# LAMMPSでトラジェクトリ生成
dump 1 all custom 100 traj.lammpstrj id type x y z

# Dissolveで読み込み・解析
# Post-processing tutorialを参照
```

---

## 6. 従来実装との比較

| 項目 | 自前実装 (Python+LAMMPS) | Dissolve |
|------|-------------------------|----------|
| EPSR精度 | ⚠️ 簡易版（勾配降下） | ✅ 正式実装 |
| EP計算 | ⚠️ `kT * delta_g` | ✅ フーリエ変換ベース |
| 信頼性 | ⚠️ 検証が必要 | ✅ 論文・コミュニティで実績 |
| 設定難易度 | 🔴 高い | 🟢 低い（GUI） |
| カスタマイズ | 🟢 自由 | ⚠️ 制限あり |

---

## 7. 次のステップ

### ステップ1: Dissolveをインストール
```bash
# パッケージをダウンロード
wget https://projectdissolve.com/packages/[your-platform]

# インストール（プラットフォーム依存）
```

### ステップ2: Argonチュートリアルを実行
```bash
# サンプルファイルをダウンロード
# GitHub releases から argon.zip を入手

# Dissolve GUI で開く
dissolve-gui argon/argon.txt
```

### ステップ3: EGaInプロジェクトを作成

Argonチュートリアルを参考に、EGaInシステム用のプロジェクトファイルを作成。

---

## 8. トラブルシューティング

### Q: 実験データのフォーマットは？

A: 2カラムのテキストファイル（スペース区切り）:
```
# 距離(Å)  g(r)
2.0  0.0
2.1  0.05
```

### Q: LAMMPSの力場パラメータを使いたい

A: Dissolveの力場ファイルに変換が必要。または、LAMMPS構造をインポート後、Dissolveの力場を再フィット。

### Q: GPUは使える？

A: DissolveはCPU並列化（OpenMP）をサポート。GPU対応は限定的。

---

## 9. 参考文献

1. **Dissolve論文**:
   Youngs, T.G.A., Marin-Rimoldi, E. and Headen, T.F. (2019)
   "Dissolve: next generation software for the interrogation of total scattering data by empirical potential generation"
   Molecular Physics, 117:22, 3464-3477
   DOI: 10.1080/00268976.2019.1651918

2. **EPSR原論文**:
   Soper, A.K. (1996)
   "Empirical potential Monte Carlo simulation of fluid structure"
   Chemical Physics, 202, 295-306

3. **EPSR User Guide**:
   ISIS Neutron and Muon Source
   https://www.isis.stfc.ac.uk/OtherFiles/Disordered Materials/EPSR26 Manual 2019-11-27.pdf

---

## 10. 結論

Dissolveを使用することで:

✅ **信頼性の高いEPSR解析** - 正式なアルゴリズム実装
✅ **時間の節約** - デバッグ・検証が不要
✅ **コミュニティサポート** - 活発な開発・ユーザーコミュニティ

**推奨**: まずDissolveのArgonチュートリアルを実行し、基本的なワークフローを理解してから、EGaInシステムに適用することをお勧めします。

---

## 連絡先・サポート

- **GitHub Issues**: https://github.com/disorderedmaterials/dissolve/issues
- **ドキュメント**: https://docs.projectdissolve.com/
- **論文著者**: Dr. Tristan Youngs (ISIS, UK)

---

**最終更新**: 2025年11月29日
