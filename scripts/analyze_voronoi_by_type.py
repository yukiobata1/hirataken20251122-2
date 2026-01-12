#!/usr/bin/env python3
"""
Type別Voronoi解析 - Ga液体の二峰性モデル解析
Type 1: 小さいGa (σ × 0.9)
Type 2: 大きいGa (σ × 1.1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from scipy import stats

# 出力ディレクトリ
VORONOI_DIR = Path("outputs/fine_search_shoulder/voronoi")
OUTPUT_DIR = Path("outputs/fine_search_shoulder/voronoi")


def read_voronoi_file(filepath: Path) -> pd.DataFrame:
    """Voronoiファイルを読み込む"""
    columns = [
        'Particle_Identifier', 'Particle_Type',
        'Position_X', 'Position_Y', 'Position_Z',
        'Velocity_X', 'Velocity_Y', 'Velocity_Z',
        'Velocity_Magnitude', 'Centrosymmetry', 'Coordination',
        'Atomic_Volume', 'Cavity_Radius',
        'Voronoi_Index_1', 'Voronoi_Index_2', 'Voronoi_Index_3',
        'Voronoi_Index_4', 'Voronoi_Index_5', 'Voronoi_Index_6',
        'Voronoi_Index_7', 'Voronoi_Index_8', 'Voronoi_Index_9',
        'Max_Face_Order'
    ]
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=2, header=None, names=columns)
    return df


def get_voronoi_signature(row) -> str:
    """Voronoiインデックスを文字列に変換"""
    indices = [int(row[f'Voronoi_Index_{i}']) for i in range(1, 10)]
    # 末尾のゼロを削除
    while indices and indices[-1] == 0:
        indices.pop()
    if not indices:
        indices = [0]
    return f"<{','.join(map(str, indices))}>"


def analyze_voronoi_by_type(all_data: dict) -> dict:
    """Type別にVoronoi多面体を分析"""

    type1_polyhedra = Counter()
    type2_polyhedra = Counter()

    for timestep, df in all_data.items():
        for _, row in df.iterrows():
            sig = get_voronoi_signature(row)
            if row['Particle_Type'] == 1:
                type1_polyhedra[sig] += 1
            else:
                type2_polyhedra[sig] += 1

    return {'type1': type1_polyhedra, 'type2': type2_polyhedra}


def plot_type_comparison(all_data: dict, output_path: Path):
    """Type別の包括的比較"""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Ga Liquid Bimodal Model: Type 1 (Small σ) vs Type 2 (Large σ)',
                 fontsize=14, fontweight='bold')

    # データ収集
    type1_coords, type2_coords = [], []
    type1_vols, type2_vols = [], []
    type1_faces, type2_faces = [], []
    type1_penta, type2_penta = [], []

    voronoi_cols = [f'Voronoi_Index_{i}' for i in range(1, 10)]

    for df in all_data.values():
        df1 = df[df['Particle_Type'] == 1]
        df2 = df[df['Particle_Type'] == 2]

        type1_coords.extend(df1['Coordination'].values)
        type2_coords.extend(df2['Coordination'].values)

        type1_vols.extend(df1['Atomic_Volume'].values)
        type2_vols.extend(df2['Atomic_Volume'].values)

        type1_faces.extend(df1[voronoi_cols].sum(axis=1).values)
        type2_faces.extend(df2[voronoi_cols].sum(axis=1).values)

        type1_penta.extend(df1['Voronoi_Index_5'].values)
        type2_penta.extend(df2['Voronoi_Index_5'].values)

    # 1. 配位数分布比較
    ax = axes[0, 0]
    bins = np.arange(8, 22) - 0.5
    ax.hist(type1_coords, bins=bins, alpha=0.6, label=f'Type 1 (Small): μ={np.mean(type1_coords):.2f}',
            color='blue', edgecolor='black', density=True)
    ax.hist(type2_coords, bins=bins, alpha=0.6, label=f'Type 2 (Large): μ={np.mean(type2_coords):.2f}',
            color='red', edgecolor='black', density=True)
    ax.set_xlabel('Coordination Number', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Coordination Number Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 2. 原子体積分布比較
    ax = axes[0, 1]
    ax.hist(type1_vols, bins=40, alpha=0.6, label=f'Type 1: μ={np.mean(type1_vols):.2f} Å³',
            color='blue', edgecolor='black', density=True)
    ax.hist(type2_vols, bins=40, alpha=0.6, label=f'Type 2: μ={np.mean(type2_vols):.2f} Å³',
            color='red', edgecolor='black', density=True)
    ax.set_xlabel('Atomic Volume (Å³)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Atomic Volume Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3. 総面数分布比較
    ax = axes[0, 2]
    bins = np.arange(8, 22) - 0.5
    ax.hist(type1_faces, bins=bins, alpha=0.6, label=f'Type 1: μ={np.mean(type1_faces):.2f}',
            color='blue', edgecolor='black', density=True)
    ax.hist(type2_faces, bins=bins, alpha=0.6, label=f'Type 2: μ={np.mean(type2_faces):.2f}',
            color='red', edgecolor='black', density=True)
    ax.set_xlabel('Total Number of Faces', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Voronoi Polyhedra Faces', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 4. 5角形面数分布比較
    ax = axes[1, 0]
    bins = np.arange(0, 15) - 0.5
    ax.hist(type1_penta, bins=bins, alpha=0.6, label=f'Type 1: μ={np.mean(type1_penta):.2f}',
            color='blue', edgecolor='black', density=True)
    ax.hist(type2_penta, bins=bins, alpha=0.6, label=f'Type 2: μ={np.mean(type2_penta):.2f}',
            color='red', edgecolor='black', density=True)
    ax.set_xlabel('Number of Pentagonal Faces', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title('Pentagonal Face Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 5. 配位数 vs 原子体積（散布図）
    ax = axes[1, 1]
    ax.scatter(type1_coords, type1_vols, alpha=0.1, s=5, c='blue', label='Type 1')
    ax.scatter(type2_coords, type2_vols, alpha=0.1, s=5, c='red', label='Type 2')

    # 平均線
    for coords, vols, color, label in [(type1_coords, type1_vols, 'blue', 'Type 1'),
                                        (type2_coords, type2_vols, 'red', 'Type 2')]:
        unique_coords = sorted(set(coords))
        mean_vols = [np.mean([v for c, v in zip(coords, vols) if c == uc]) for uc in unique_coords]
        ax.plot(unique_coords, mean_vols, color=color, linewidth=2, marker='o', markersize=5)

    ax.set_xlabel('Coordination Number', fontsize=11)
    ax.set_ylabel('Atomic Volume (Å³)', fontsize=11)
    ax.set_title('Coordination vs Volume', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 6. 統計サマリー
    ax = axes[1, 2]
    ax.axis('off')

    # 統計検定
    t_coord, p_coord = stats.ttest_ind(type1_coords, type2_coords)
    t_vol, p_vol = stats.ttest_ind(type1_vols, type2_vols)
    t_faces, p_faces = stats.ttest_ind(type1_faces, type2_faces)
    t_penta, p_penta = stats.ttest_ind(type1_penta, type2_penta)

    summary = f"""
    ══════════════════════════════════════════
         Type Comparison Summary
    ══════════════════════════════════════════

    Type 1 (Small σ): {len(type1_coords)//len(all_data)} atoms/frame
    Type 2 (Large σ): {len(type2_coords)//len(all_data)} atoms/frame

    ──────────────────────────────────────────
                    Type 1      Type 2     p-value
    ──────────────────────────────────────────
    Coordination   {np.mean(type1_coords):6.2f}      {np.mean(type2_coords):6.2f}     {p_coord:.2e}
    Volume (Å³)    {np.mean(type1_vols):6.2f}      {np.mean(type2_vols):6.2f}     {p_vol:.2e}
    Total Faces    {np.mean(type1_faces):6.2f}      {np.mean(type2_faces):6.2f}     {p_faces:.2e}
    Pentagon Faces {np.mean(type1_penta):6.2f}      {np.mean(type2_penta):6.2f}     {p_penta:.2e}
    ──────────────────────────────────────────

    Interpretation:
    • Type 2 (Large σ) has {"higher" if np.mean(type2_vols) > np.mean(type1_vols) else "lower"} atomic volume
    • Type 2 has {"more" if np.mean(type2_coords) > np.mean(type1_coords) else "fewer"} neighbors
    • p < 0.05 indicates significant difference

    ══════════════════════════════════════════
    """

    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_polyhedra_by_type(all_data: dict, output_path: Path):
    """Type別のVoronoi多面体分布"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Voronoi Polyhedra Distribution by Type', fontsize=14, fontweight='bold')

    # Type別の多面体カウント
    polyhedra_data = analyze_voronoi_by_type(all_data)
    type1_counter = polyhedra_data['type1']
    type2_counter = polyhedra_data['type2']

    # 全多面体の合計
    all_counter = type1_counter + type2_counter
    top20 = [p[0] for p in all_counter.most_common(20)]

    # 1. Type 1の上位多面体
    ax = axes[0, 0]
    top1 = type1_counter.most_common(15)
    labels = [p[0] for p in top1]
    counts = [p[1] for p in top1]
    total = sum(type1_counter.values())
    percentages = [c / total * 100 for c in counts]

    bars = ax.barh(range(len(labels)), percentages, color='blue', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Frequency (%)', fontsize=11)
    ax.set_title('Type 1 (Small σ) - Top 15 Polyhedra', fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 2. Type 2の上位多面体
    ax = axes[0, 1]
    top2 = type2_counter.most_common(15)
    labels = [p[0] for p in top2]
    counts = [p[1] for p in top2]
    total = sum(type2_counter.values())
    percentages = [c / total * 100 for c in counts]

    bars = ax.barh(range(len(labels)), percentages, color='red', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Frequency (%)', fontsize=11)
    ax.set_title('Type 2 (Large σ) - Top 15 Polyhedra', fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 3. 上位20多面体のType別比較
    ax = axes[1, 0]
    x = np.arange(len(top20))
    width = 0.35

    total1 = sum(type1_counter.values())
    total2 = sum(type2_counter.values())

    pct1 = [type1_counter.get(p, 0) / total1 * 100 for p in top20]
    pct2 = [type2_counter.get(p, 0) / total2 * 100 for p in top20]

    ax.bar(x - width/2, pct1, width, label='Type 1 (Small)', color='blue', alpha=0.7)
    ax.bar(x + width/2, pct2, width, label='Type 2 (Large)', color='red', alpha=0.7)
    ax.set_xlabel('Polyhedra Type', fontsize=11)
    ax.set_ylabel('Frequency (%)', fontsize=11)
    ax.set_title('Top 20 Polyhedra Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(top20, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Type間の差分（どの多面体がType特異的か）
    ax = axes[1, 1]

    # 差分を計算
    diff = {}
    for p in set(list(type1_counter.keys()) + list(type2_counter.keys())):
        pct1 = type1_counter.get(p, 0) / total1 * 100
        pct2 = type2_counter.get(p, 0) / total2 * 100
        diff[p] = pct2 - pct1  # 正: Type 2に多い、負: Type 1に多い

    # 差が大きい上位10ずつ
    sorted_diff = sorted(diff.items(), key=lambda x: x[1])
    type1_specific = sorted_diff[:10]  # Type 1に多い
    type2_specific = sorted_diff[-10:][::-1]  # Type 2に多い

    combined = type1_specific + type2_specific
    labels = [p[0] for p in combined]
    values = [p[1] for p in combined]
    colors = ['blue' if v < 0 else 'red' for v in values]

    ax.barh(range(len(labels)), values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Δ Frequency (%) [Type 2 - Type 1]', fontsize=11)
    ax.set_title('Type-Specific Polyhedra\n(Blue: Type 1 enriched, Red: Type 2 enriched)', fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_local_structure_analysis(all_data: dict, output_path: Path):
    """局所構造の詳細解析"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Local Structure Analysis: Shoulder Structure Origin',
                 fontsize=14, fontweight='bold')

    voronoi_cols = [f'Voronoi_Index_{i}' for i in range(1, 10)]

    # データ収集
    data_by_type = {1: [], 2: []}
    for df in all_data.values():
        for t in [1, 2]:
            df_t = df[df['Particle_Type'] == t]
            for _, row in df_t.iterrows():
                data_by_type[t].append({
                    'coord': row['Coordination'],
                    'vol': row['Atomic_Volume'],
                    'penta': row['Voronoi_Index_5'],
                    'hexa': row['Voronoi_Index_6'],
                    'total_faces': sum(row[voronoi_cols]),
                })

    # 1. 5角形 vs 6角形（Type別）
    ax = axes[0, 0]
    for t, color, label in [(1, 'blue', 'Type 1 (Small)'), (2, 'red', 'Type 2 (Large)')]:
        penta = [d['penta'] for d in data_by_type[t]]
        hexa = [d['hexa'] for d in data_by_type[t]]
        ax.scatter(penta, hexa, alpha=0.05, s=10, c=color, label=label)

    # 平均点
    for t, color in [(1, 'blue'), (2, 'red')]:
        mean_p = np.mean([d['penta'] for d in data_by_type[t]])
        mean_h = np.mean([d['hexa'] for d in data_by_type[t]])
        ax.scatter(mean_p, mean_h, s=200, c=color, marker='*', edgecolor='black', linewidth=2)

    ax.set_xlabel('Number of Pentagonal Faces', fontsize=11)
    ax.set_ylabel('Number of Hexagonal Faces', fontsize=11)
    ax.set_title('Pentagon vs Hexagon Faces\n(Stars = Mean)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 2. 配位数の時間変化（Type別）
    ax = axes[0, 1]
    timesteps = sorted(all_data.keys())

    for t, color, label in [(1, 'blue', 'Type 1'), (2, 'red', 'Type 2')]:
        means = []
        stds = []
        for ts in timesteps:
            df_t = all_data[ts][all_data[ts]['Particle_Type'] == t]
            means.append(df_t['Coordination'].mean())
            stds.append(df_t['Coordination'].std())
        ax.errorbar(timesteps, means, yerr=stds, marker='o', capsize=3,
                   color=color, label=label, linewidth=2, markersize=5)

    ax.set_xlabel('Timestep (×1000)', fontsize=11)
    ax.set_ylabel('Mean Coordination Number', fontsize=11)
    ax.set_title('Coordination vs Time by Type', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3. 原子体積の時間変化（Type別）
    ax = axes[1, 0]
    for t, color, label in [(1, 'blue', 'Type 1'), (2, 'red', 'Type 2')]:
        means = []
        stds = []
        for ts in timesteps:
            df_t = all_data[ts][all_data[ts]['Particle_Type'] == t]
            means.append(df_t['Atomic_Volume'].mean())
            stds.append(df_t['Atomic_Volume'].std())
        ax.errorbar(timesteps, means, yerr=stds, marker='o', capsize=3,
                   color=color, label=label, linewidth=2, markersize=5)

    ax.set_xlabel('Timestep (×1000)', fontsize=11)
    ax.set_ylabel('Mean Atomic Volume (Å³)', fontsize=11)
    ax.set_title('Atomic Volume vs Time by Type', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 4. 肩構造の解釈（模式図＋説明）
    ax = axes[1, 1]
    ax.axis('off')

    interpretation = """
    ══════════════════════════════════════════════════════
          Interpretation: Origin of Shoulder Structure
    ══════════════════════════════════════════════════════

    The bimodal model (Type 1 + Type 2) captures the
    heterogeneous local structure of liquid Ga:

    ┌─────────────────┬─────────────────┐
    │    Type 1       │    Type 2       │
    │   (Small σ)     │   (Large σ)     │
    ├─────────────────┼─────────────────┤
    │ • Lower volume  │ • Higher volume │
    │ • More compact  │ • More open     │
    │ • Higher coord  │ • Lower coord   │
    │   (locally)     │   (locally)     │
    └─────────────────┴─────────────────┘

    → The coexistence of these two local environments
      creates the "shoulder" in S(Q) and g(r)

    → This reflects the remnant of Ga's covalent
      bonding character in the liquid state

    → The pentagonal faces indicate icosahedral-like
      short-range order (ISRO), which is common in
      liquid metals and metallic glasses

    ══════════════════════════════════════════════════════
    """

    ax.text(0.05, 0.5, interpretation, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """メイン処理"""

    print("=" * 60)
    print("Type-Specific Voronoi Analysis")
    print("Ga Liquid Bimodal Model")
    print("=" * 60)

    # voronoiファイルを検索
    voronoi_files = sorted(VORONOI_DIR.glob("voronoi.*"),
                           key=lambda x: int(x.suffix[1:]) if x.suffix[1:].isdigit() else 0)

    if not voronoi_files:
        print(f"No voronoi files found in {VORONOI_DIR}")
        return

    print(f"Found {len(voronoi_files)} voronoi files")

    # 全ファイルを読み込み
    all_data = {}
    for f in voronoi_files:
        timestep = int(f.suffix[1:]) if f.suffix[1:].isdigit() else 0
        df = read_voronoi_file(f)
        all_data[timestep] = df

    # Type別の原子数を確認
    df0 = list(all_data.values())[0]
    n_type1 = len(df0[df0['Particle_Type'] == 1])
    n_type2 = len(df0[df0['Particle_Type'] == 2])
    print(f"\nAtom counts:")
    print(f"  Type 1 (Small σ): {n_type1}")
    print(f"  Type 2 (Large σ): {n_type2}")

    print("\nGenerating Type-specific plots...")

    # グラフ作成
    plot_type_comparison(all_data, OUTPUT_DIR / "voronoi_type_comparison.png")
    plot_polyhedra_by_type(all_data, OUTPUT_DIR / "voronoi_polyhedra_by_type.png")
    plot_local_structure_analysis(all_data, OUTPUT_DIR / "voronoi_local_structure.png")

    print("\n" + "=" * 60)
    print("Done! Output files:")
    for f in sorted(OUTPUT_DIR.glob("voronoi_*.png")):
        print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
