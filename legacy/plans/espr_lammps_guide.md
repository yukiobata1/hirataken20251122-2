# LAMMPSでEPSR法を実装する手順書

**EGaIn液体金属シミュレーション向け**

---

## 1. 概要

EPSR（Empirical Potential Structure Refinement）法は、実験データ（回折データなど）とシミュレーションを組み合わせて原子構造モデルを構築する手法です。本ドキュメントでは、LAMMPSとPythonを使用してEPSR法を実装する手順を説明します。

### 1.1 基本原理

EPSRの核心は、以下の式に基づいています：

$$U_{total}(r) = U_{LJ}(r) + U_{EP}(r)$$

- **U_LJ(r)**：初期ポテンシャル（Lennard-Jonesなど）
- **U_EP(r)**：経験的補正ポテンシャル（実験データとの差を埋める）

### 1.2 必要なもの

- LAMMPS（分子動力学シミュレーション）
- Python 3（NumPy, SciPy, Matplotlib）
- 実験データ（g(r)またはS(Q)）

---

## 2. ワークフロー全体像

| Step | 内容 |
|------|------|
| 1 | 初期構造とLJポテンシャルの設定 |
| 2 | LAMMPSでMCシミュレーション実行 |
| 3 | g(r)の計算と実験データとの比較 |
| 4 | U_EP(r)の更新 |
| 5 | 収束判定（収束していなければStep 2へ戻る） |

```
┌─────────────────────────────────────────────────────────┐
│                    初期化                                │
│  • LJパラメータ設定                                      │
│  • U_EP = 0 で初期化                                    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              LAMMPSシミュレーション                       │
│  • U_total = U_LJ + U_EP でMD/MC実行                    │
│  • g_sim(r) を計算                                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  比較・評価                              │
│  • χ² = Σ[g_sim(r) - g_exp(r)]² を計算                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                    ┌──────────┐
                    │ χ² < tol │───Yes──→ 終了
                    └──────────┘
                          │ No
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  U_EP更新                               │
│  • U_EP += α × kT × (g_sim - g_exp)                    │
└─────────────────────────────────────────────────────────┘
                          │
                          └──────→ LAMMPSシミュレーションへ戻る
```

---

## 3. 初期設定

### 3.1 EGaInのLJパラメータ

Amon et al. (2023)の論文より、以下のパラメータを使用します：

| 原子 | σ (Å) | ε (kJ/mol) | ε (kcal/mol) |
|------|-------|------------|--------------|
| Ga | 2.70 | 1.80 | 0.430 |
| In | 3.11 | 1.80 | 0.430 |

Ga-In間の相互作用は混合則（Lorentz-Berthelot則）で計算します：

$$\sigma_{GaIn} = \frac{\sigma_{Ga} + \sigma_{In}}{2} = 2.905 \text{ Å}$$

$$\varepsilon_{GaIn} = \sqrt{\varepsilon_{Ga} \times \varepsilon_{In}} = 1.80 \text{ kJ/mol}$$

### 3.2 初期構造の作成

共晶組成 Ga₀.₈₅₈In₀.₁₄₂ のランダム構造を作成します：

```python
# create_initial_structure.py
import numpy as np

n_atoms = 1000
x_In = 0.142  # インジウムのモル分率
n_In = int(n_atoms * x_In)
n_Ga = n_atoms - n_In

# 実験密度から箱サイズを計算
rho = 6.28  # g/cm³ (150°Cでの密度)
M_Ga, M_In = 69.723, 114.818  # g/mol
M_avg = (1 - x_In) * M_Ga + x_In * M_In
V = n_atoms * M_avg / (rho * 6.022e23) * 1e24  # Å³
L = V ** (1/3)  # 立方体の一辺

print(f"Box size: {L:.2f} Å")
print(f"Number of atoms: Ga={n_Ga}, In={n_In}")

# ランダム配置
positions = np.random.rand(n_atoms, 3) * L
atom_types = [1] * n_Ga + [2] * n_In

# LAMMPS data file出力
with open('initial_structure.data', 'w') as f:
    f.write('EGaIn initial structure\n\n')
    f.write(f'{n_atoms} atoms\n')
    f.write('2 atom types\n\n')
    f.write(f'0.0 {L:.6f} xlo xhi\n')
    f.write(f'0.0 {L:.6f} ylo yhi\n')
    f.write(f'0.0 {L:.6f} zlo zhi\n\n')
    f.write('Masses\n\n')
    f.write('1 69.723\n')
    f.write('2 114.818\n\n')
    f.write('Atoms\n\n')
    for i, (pos, atype) in enumerate(zip(positions, atom_types)):
        f.write(f'{i+1} {atype} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n')
```

---

## 4. LAMMPSスクリプト

### 4.1 基本シミュレーション（LJのみ）

```lammps
# in.egain_lj
units real
atom_style atomic
boundary p p p

read_data initial_structure.data

# LJパラメータ (ε: kcal/mol, σ: Å)
# 1.80 kJ/mol = 0.430 kcal/mol
pair_style lj/cut 12.0
pair_coeff 1 1 0.430 2.70   # Ga-Ga
pair_coeff 2 2 0.430 3.11   # In-In
pair_coeff 1 2 0.430 2.905  # Ga-In

mass 1 69.723   # Ga
mass 2 114.818  # In

# 温度設定 (150°C = 423.15 K)
velocity all create 423.15 12345

# NVT平衡化
fix 1 all nvt temp 423.15 423.15 100.0
timestep 2.0

# g(r)計算のための設定
# rdf: Nbins, type1-type1, type1-type2, type2-type2
compute rdf all rdf 200 1 1 1 2 2 2
fix 2 all ave/time 100 10 1000 c_rdf[*] file rdf.dat mode vector

# 実行
thermo 1000
thermo_style custom step temp pe ke etotal press

# 平衡化
run 50000

# サンプリング
run 50000
```

### 4.2 テーブルポテンシャルを使用する場合（U_EP追加）

```lammps
# in.egain_epsr
units real
atom_style atomic
boundary p p p

read_data initial_structure.data

# LJ + EPテーブルを使用
pair_style hybrid/overlay lj/cut 12.0 table linear 1000

# LJポテンシャル
pair_coeff 1 1 lj/cut 0.430 2.70
pair_coeff 2 2 lj/cut 0.430 3.11
pair_coeff 1 2 lj/cut 0.430 2.905

# 経験的ポテンシャル（テーブル形式）
pair_coeff 1 1 table ep_GaGa.table EP_GAGA
pair_coeff 2 2 table ep_InIn.table EP_ININ
pair_coeff 1 2 table ep_GaIn.table EP_GAIN

mass 1 69.723
mass 2 114.818

velocity all create 423.15 12345
fix 1 all nvt temp 423.15 423.15 100.0
timestep 2.0

compute rdf all rdf 200 1 1 1 2 2 2
fix 2 all ave/time 100 10 1000 c_rdf[*] file rdf.dat mode vector

thermo 1000
run 100000
```

---

## 5. U_EP更新アルゴリズム

### 5.1 更新式

各反復でU_EPを以下のように更新します：

$$U_{EP}^{(n+1)}(r) = U_{EP}^{(n)}(r) + \alpha \cdot k_B T \cdot [g_{sim}(r) - g_{exp}(r)]$$

- **α**：学習率（0.1〜0.5程度）
- **k_B × T**：熱エネルギー（423.15 Kで約 0.84 kcal/mol）

### 5.2 Pythonスクリプト

```python
# update_ep.py
import numpy as np

def update_ep(r, g_sim, g_exp, U_ep_old, alpha=0.3, T=423.15):
    """
    U_EPを更新する
    
    Parameters
    ----------
    r : array
        距離配列 (Å)
    g_sim : array
        シミュレーションのg(r)
    g_exp : array
        実験のg(r)
    U_ep_old : array
        現在のU_EP (kcal/mol)
    alpha : float
        学習率
    T : float
        温度 (K)
    
    Returns
    -------
    U_ep_new : array
        更新後のU_EP (kcal/mol)
    """
    kB = 0.001987  # kcal/(mol·K)
    kT = kB * T
    
    # 差分を計算
    delta_g = g_sim - g_exp
    
    # U_EPを更新
    U_ep_new = U_ep_old + alpha * kT * delta_g
    
    # 振幅制限（オプション）
    max_amp = 1.0  # kcal/mol
    U_ep_new = np.clip(U_ep_new, -max_amp, max_amp)
    
    # スムージング（オプション）
    from scipy.ndimage import gaussian_filter1d
    U_ep_new = gaussian_filter1d(U_ep_new, sigma=2)
    
    return U_ep_new
```

### 5.3 LAMMPSテーブルファイルの生成

```python
def write_lammps_table(filename, r, U_ep, label):
    """
    LAMMPSテーブルファイルを書き出す
    
    Parameters
    ----------
    filename : str
        出力ファイル名
    r : array
        距離配列 (Å)
    U_ep : array
        ポテンシャル (kcal/mol)
    label : str
        テーブルのラベル名
    """
    # 力の計算 (F = -dU/dr)
    F = -np.gradient(U_ep, r)
    
    with open(filename, 'w') as f:
        f.write(f'# Empirical potential table\n')
        f.write(f'# Generated by EPSR iteration\n\n')
        f.write(f'{label}\n')
        f.write(f'N {len(r)}\n\n')
        
        for i, (ri, ui, fi) in enumerate(zip(r, U_ep, F)):
            f.write(f'{i+1} {ri:.6f} {ui:.6f} {fi:.6f}\n')


# 使用例
r = np.linspace(2.0, 12.0, 200)
U_ep = np.zeros_like(r)  # 初期値はゼロ

write_lammps_table('ep_GaGa.table', r, U_ep, 'EP_GAGA')
write_lammps_table('ep_InIn.table', r, U_ep, 'EP_ININ')
write_lammps_table('ep_GaIn.table', r, U_ep, 'EP_GAIN')
```

---

## 6. 収束判定

### 6.1 χ²の計算

収束判定には、以下のχ²を使用します：

$$\chi^2 = \sum_i \frac{[g_{sim}(r_i) - g_{exp}(r_i)]^2}{\sigma_i^2}$$

```python
def calc_chi_squared(g_sim, g_exp, sigma=0.01):
    """
    χ²を計算
    
    Parameters
    ----------
    g_sim : array
        シミュレーションのg(r)
    g_exp : array
        実験のg(r)
    sigma : float or array
        測定の標準偏差
    
    Returns
    -------
    chi2 : float
        χ²値
    """
    return np.sum((g_sim - g_exp)**2 / sigma**2)


def calc_r_factor(g_sim, g_exp):
    """
    R-factorを計算（結晶学でよく使われる指標）
    """
    return np.sum(np.abs(g_sim - g_exp)) / np.sum(np.abs(g_exp))
```

### 6.2 収束条件

- χ²が十分小さくなった（例：χ² < 0.1）
- χ²の変化が小さくなった（例：Δχ² / χ² < 0.01）
- 最大反復回数に達した（例：100回）

---

## 7. メインループの実装

```python
# main_epsr.py
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def read_lammps_rdf(filename):
    """LAMMPSのRDF出力を読み込む"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    data.append([float(x) for x in parts])
                except:
                    continue
    
    data = np.array(data)
    # 列: index, r, g_total, g_11, g_12, g_22, ...
    r = data[:, 1]
    g_total = data[:, 2]
    return r, g_total


def load_experimental_data(filename):
    """実験データを読み込む"""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]  # r, g(r)


def run_lammps(input_file):
    """LAMMPSを実行"""
    result = subprocess.run(
        ['lmp', '-in', input_file],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"LAMMPS error: {result.stderr}")
        raise RuntimeError("LAMMPS failed")


def main():
    # パラメータ
    max_iter = 50
    alpha = 0.3
    tol = 0.1
    T = 423.15  # K
    
    # 距離グリッド
    r = np.linspace(2.0, 12.0, 200)
    
    # 実験データ読み込み
    r_exp, g_exp = load_experimental_data('g_exp.dat')
    
    # U_EPの初期化（ゼロ、各ペアごと）
    U_ep_GaGa = np.zeros_like(r)
    U_ep_InIn = np.zeros_like(r)
    U_ep_GaIn = np.zeros_like(r)
    
    # 収束履歴
    chi2_history = []
    
    for iteration in range(max_iter):
        print(f"\n=== Iteration {iteration + 1} ===")
        
        # 1. LAMMPSテーブルファイル生成
        write_lammps_table('ep_GaGa.table', r, U_ep_GaGa, 'EP_GAGA')
        write_lammps_table('ep_InIn.table', r, U_ep_InIn, 'EP_ININ')
        write_lammps_table('ep_GaIn.table', r, U_ep_GaIn, 'EP_GAIN')
        
        # 2. LAMMPSシミュレーション実行
        run_lammps('in.egain_epsr')
        
        # 3. g(r)を読み込み
        r_sim, g_sim = read_lammps_rdf('rdf.dat')
        
        # 実験データと同じグリッドに補間
        g_sim_interp = np.interp(r_exp, r_sim, g_sim)
        
        # 4. χ²を計算
        chi2 = calc_chi_squared(g_sim_interp, g_exp)
        chi2_history.append(chi2)
        print(f"χ² = {chi2:.4f}")
        
        # 5. 収束判定
        if chi2 < tol:
            print("Converged!")
            break
        
        if len(chi2_history) > 1:
            delta_chi2 = abs(chi2_history[-1] - chi2_history[-2])
            if delta_chi2 / chi2 < 0.01:
                print("χ² change is small. Converged!")
                break
        
        # 6. U_EPを更新（簡略化：全体のg(r)で更新）
        # 本来は部分g(r)ごとに更新すべき
        g_exp_interp = np.interp(r, r_exp, g_exp)
        g_sim_on_r = np.interp(r, r_sim, g_sim)
        
        U_ep_GaGa = update_ep(r, g_sim_on_r, g_exp_interp, U_ep_GaGa, alpha, T)
        U_ep_InIn = update_ep(r, g_sim_on_r, g_exp_interp, U_ep_InIn, alpha, T)
        U_ep_GaIn = update_ep(r, g_sim_on_r, g_exp_interp, U_ep_GaIn, alpha, T)
    
    # 結果をプロット
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # g(r)の比較
    axes[0].plot(r_exp, g_exp, 'k-', label='Experiment', linewidth=2)
    axes[0].plot(r_sim, g_sim, 'r--', label='Simulation', linewidth=2)
    axes[0].set_xlabel('r (Å)')
    axes[0].set_ylabel('g(r)')
    axes[0].legend()
    axes[0].set_title('Pair Distribution Function')
    
    # U_EPの形状
    axes[1].plot(r, U_ep_GaGa, label='Ga-Ga')
    axes[1].plot(r, U_ep_InIn, label='In-In')
    axes[1].plot(r, U_ep_GaIn, label='Ga-In')
    axes[1].set_xlabel('r (Å)')
    axes[1].set_ylabel('U_EP (kcal/mol)')
    axes[1].legend()
    axes[1].set_title('Empirical Potential')
    
    # 収束履歴
    axes[2].plot(chi2_history, 'o-')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('χ²')
    axes[2].set_title('Convergence')
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('epsr_results.png', dpi=150)
    plt.show()
    
    # 最終的なU_EPを保存
    np.savez('final_ep.npz', r=r, 
             U_ep_GaGa=U_ep_GaGa, 
             U_ep_InIn=U_ep_InIn, 
             U_ep_GaIn=U_ep_GaIn)


if __name__ == '__main__':
    main()
```

---

## 8. 実装のヒント

### 8.1 よくある問題と対処法

| 問題 | 対処法 |
|------|--------|
| 収束が遅い | αを大きくする（ただし不安定になる可能性） |
| 振動して収束しない | αを小さくする、または振幅制限を厳しくする |
| U_EPが非物理的 | 振幅制限（max_amp）を設定、スムージングを適用 |
| シミュレーションが不安定 | タイムステップを小さくする、U_EPの変化を緩やかに |
| 短距離でg(r)がずれる | r < 2.5 Åの範囲は除外して評価する |

### 8.2 推奨パラメータ

- **学習率 α**：0.1〜0.5（最初は小さめで始める）
- **振幅制限**：1〜4 kJ/mol（≈ 0.24〜0.96 kcal/mol）
- **平衡化ステップ**：各反復で50,000〜100,000ステップ
- **g(r)サンプリング**：平衡化後に10,000ステップ以上
- **グリッド間隔**：0.05 Å程度

### 8.3 部分g(r)を使う場合

より正確な結果を得るには、全体のg(r)ではなく部分g(r)（Ga-Ga, Ga-In, In-In）ごとにU_EPを更新します：

```python
# 部分g(r)の読み込み（LAMMPSの出力形式による）
# compute rdf all rdf 200 1 1 1 2 2 2 の場合
# 列: index, r, g_total, coord_total, g_11, coord_11, g_12, coord_12, g_22, coord_22

def read_partial_rdf(filename):
    """部分RDFを読み込む"""
    data = np.loadtxt(filename, comments='#')
    r = data[:, 1]
    g_GaGa = data[:, 4]   # 1-1
    g_GaIn = data[:, 6]   # 1-2
    g_InIn = data[:, 8]   # 2-2
    return r, g_GaGa, g_GaIn, g_InIn
```

---

## 9. ファイル構成

最終的なディレクトリ構成：

```
egain_epsr/
├── create_initial_structure.py  # 初期構造作成
├── initial_structure.data       # LAMMPS data file
├── in.egain_lj                  # LJのみのLAMMPSスクリプト
├── in.egain_epsr                # EPSR用LAMMPSスクリプト
├── update_ep.py                 # U_EP更新関数
├── main_epsr.py                 # メインループ
├── g_exp.dat                    # 実験データ
├── ep_GaGa.table                # Ga-GaのU_EPテーブル
├── ep_InIn.table                # In-InのU_EPテーブル
├── ep_GaIn.table                # Ga-InのU_EPテーブル
├── rdf.dat                      # LAMMPSからのRDF出力
├── final_ep.npz                 # 最終的なU_EP
└── epsr_results.png             # 結果のプロット
```

---

## 10. 参考文献

1. Amon, A. et al. (2023). Local Order in Liquid Gallium–Indium Alloys. *J. Phys. Chem. C*, 127(33), 16687-16694. https://doi.org/10.1021/acs.jpcc.3c03857

2. Soper, A. K. (1996). Empirical potential Monte Carlo simulation of fluid structure. *Chem. Phys.*, 202, 295-306.

3. Soper, A. K. (2005). Partial structure factors from disordered materials diffraction data: An approach using empirical potential structure refinement. *Phys. Rev. B*, 72, 104204.

4. LAMMPS Documentation: https://docs.lammps.org/

---

## 付録：実験データのフォーマット

実験データ `g_exp.dat` は以下の形式を想定：

```
# r (Å)    g(r)
2.50       0.05
2.55       0.12
2.60       0.25
...
```

論文のSupporting Informationや、放射光施設のデータベースから取得できる場合があります。
