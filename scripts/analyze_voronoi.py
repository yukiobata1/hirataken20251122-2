#!/usr/bin/env python3
"""
Voronoi解析結果のグラフ作成スクリプト
voronoi.{i}ファイルを解析してグラフ化する
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import re

# 出力ディレクトリ
VORONOI_DIR = Path("outputs/fine_search_shoulder/voronoi")
OUTPUT_DIR = Path("outputs/fine_search_shoulder/voronoi")

def read_voronoi_file(filepath: Path) -> pd.DataFrame:
    """Voronoiファイルを読み込む"""

    # 固定のカラム名を使用
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

    # データを読み込み（最初の2行はヘッダーなのでスキップ）
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=2, header=None, names=columns)

    return df


def get_voronoi_indices(df: pd.DataFrame) -> np.ndarray:
    """Voronoiインデックスを抽出 (面の数の分布)"""

    voronoi_cols = [c for c in df.columns if 'Voronoi_Index' in c]
    return df[voronoi_cols].values


def create_voronoi_index_string(indices: np.ndarray) -> str:
    """Voronoiインデックスを文字列に変換 (例: <0,0,12,0>)"""
    # 先頭のゼロをスキップし、最後の連続するゼロを削除
    non_zero_end = len(indices)
    for i in range(len(indices) - 1, -1, -1):
        if indices[i] != 0:
            non_zero_end = i + 1
            break

    return f"<{','.join(map(str, indices[:non_zero_end]))}>"


def analyze_voronoi_polyhedra(df: pd.DataFrame) -> dict:
    """Voronoi多面体の種類を分析"""

    voronoi_cols = [c for c in df.columns if 'Voronoi_Index' in c]
    indices = df[voronoi_cols].values.astype(int)

    # 各多面体の種類をカウント
    polyhedra_counter = Counter()
    for idx in indices:
        key = tuple(idx)
        polyhedra_counter[key] += 1

    return polyhedra_counter


def plot_coordination_distribution(all_data: dict, output_path: Path):
    """配位数分布をプロット"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 全時間平均
    ax = axes[0, 0]
    all_coords = []
    for timestep, df in all_data.items():
        all_coords.extend(df['Coordination'].values)

    coords, counts = np.unique(all_coords, return_counts=True)
    ax.bar(coords, counts / len(all_coords) * 100, color='steelblue', edgecolor='black')
    ax.set_xlabel('Coordination Number', fontsize=12)
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_title('Average Coordination Number Distribution', fontsize=14)
    ax.set_xticks(range(int(coords.min()), int(coords.max()) + 1))
    ax.grid(axis='y', alpha=0.3)

    # Type別
    ax = axes[0, 1]
    type1_coords = []
    type2_coords = []
    for timestep, df in all_data.items():
        type1_coords.extend(df[df['Particle_Type'] == 1]['Coordination'].values)
        type2_coords.extend(df[df['Particle_Type'] == 2]['Coordination'].values)

    coords1, counts1 = np.unique(type1_coords, return_counts=True)
    coords2, counts2 = np.unique(type2_coords, return_counts=True)

    width = 0.35
    ax.bar(coords1 - width/2, counts1 / len(type1_coords) * 100, width,
           label='Ga (Type 1)', color='coral', edgecolor='black')
    ax.bar(coords2 + width/2, counts2 / len(type2_coords) * 100, width,
           label='In (Type 2)', color='skyblue', edgecolor='black')
    ax.set_xlabel('Coordination Number', fontsize=12)
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_title('Coordination by Atom Type', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 時間変化（平均配位数）
    ax = axes[1, 0]
    timesteps = sorted(all_data.keys())
    mean_coords = [all_data[t]['Coordination'].mean() for t in timesteps]
    std_coords = [all_data[t]['Coordination'].std() for t in timesteps]

    ax.errorbar(timesteps, mean_coords, yerr=std_coords, marker='o',
                capsize=3, color='steelblue', linewidth=2, markersize=6)
    ax.set_xlabel('Timestep (x1000)', fontsize=12)
    ax.set_ylabel('Mean Coordination Number', fontsize=12)
    ax.set_title('Coordination Number vs Time', fontsize=14)
    ax.grid(alpha=0.3)

    # 配位数のヒートマップ（時間変化）
    ax = axes[1, 1]
    coord_range = range(10, 20)
    time_coord_matrix = np.zeros((len(timesteps), len(coord_range)))

    for i, t in enumerate(timesteps):
        coords, counts = np.unique(all_data[t]['Coordination'].values, return_counts=True)
        total = len(all_data[t])
        for c, n in zip(coords, counts):
            if int(c) in coord_range:
                time_coord_matrix[i, int(c) - 10] = n / total * 100

    im = ax.imshow(time_coord_matrix.T, aspect='auto', origin='lower',
                   cmap='YlOrRd', extent=[timesteps[0], timesteps[-1], 10, 19])
    ax.set_xlabel('Timestep (x1000)', fontsize=12)
    ax.set_ylabel('Coordination Number', fontsize=12)
    ax.set_title('Coordination Distribution Over Time', fontsize=14)
    plt.colorbar(im, ax=ax, label='Frequency (%)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_voronoi_polyhedra(all_data: dict, output_path: Path):
    """Voronoi多面体分布をプロット"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 全多面体のカウント
    total_counter = Counter()
    for timestep, df in all_data.items():
        counter = analyze_voronoi_polyhedra(df)
        total_counter.update(counter)

    # 上位20の多面体
    top_polyhedra = total_counter.most_common(20)

    ax = axes[0, 0]
    labels = [f"<{','.join(map(str, [x for x in p[0] if x != 0 or p[0].index(x) < 4]))}>".replace('<>', '<0>') for p in top_polyhedra]
    # 簡略化したラベル
    labels = []
    for p in top_polyhedra:
        idx = list(p[0])
        # 末尾のゼロを削除
        while idx and idx[-1] == 0:
            idx.pop()
        if not idx:
            idx = [0]
        labels.append(f"<{','.join(map(str, idx))}>")

    counts = [p[1] for p in top_polyhedra]
    total = sum(total_counter.values())
    percentages = [c / total * 100 for c in counts]

    bars = ax.barh(range(len(labels)), percentages, color='teal', edgecolor='black')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Frequency (%)', fontsize=12)
    ax.set_title('Top 20 Voronoi Polyhedra', fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 面の総数の分布
    ax = axes[0, 1]
    total_faces_all = []
    for timestep, df in all_data.items():
        voronoi_cols = [c for c in df.columns if 'Voronoi_Index' in c]
        total_faces = df[voronoi_cols].sum(axis=1).values
        total_faces_all.extend(total_faces)

    faces, counts = np.unique(total_faces_all, return_counts=True)
    ax.bar(faces, counts / len(total_faces_all) * 100, color='purple', edgecolor='black')
    ax.set_xlabel('Total Number of Faces', fontsize=12)
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_title('Voronoi Polyhedra Total Faces Distribution', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # 5角形面と6角形面の分布
    ax = axes[1, 0]
    penta_all = []
    hexa_all = []
    for timestep, df in all_data.items():
        # Index_5 = 5角形面, Index_6 = 6角形面
        penta = df['Voronoi_Index_5'].values
        hexa = df['Voronoi_Index_6'].values
        penta_all.extend(penta)
        hexa_all.extend(hexa)

    ax.hist2d(penta_all, hexa_all, bins=[range(0, 15), range(0, 12)],
              cmap='YlGnBu', cmin=1)
    ax.set_xlabel('Number of Pentagonal Faces', fontsize=12)
    ax.set_ylabel('Number of Hexagonal Faces', fontsize=12)
    ax.set_title('Pentagon vs Hexagon Face Distribution', fontsize=14)
    plt.colorbar(ax.collections[0], ax=ax, label='Count')

    # 多面体の時間変化（上位5種）
    ax = axes[1, 1]
    timesteps = sorted(all_data.keys())
    top5_keys = [p[0] for p in top_polyhedra[:5]]

    for key in top5_keys:
        idx = list(key)
        while idx and idx[-1] == 0:
            idx.pop()
        label = f"<{','.join(map(str, idx))}>"

        fractions = []
        for t in timesteps:
            counter = analyze_voronoi_polyhedra(all_data[t])
            total = sum(counter.values())
            fractions.append(counter.get(key, 0) / total * 100)

        ax.plot(timesteps, fractions, marker='o', label=label, linewidth=2, markersize=5)

    ax.set_xlabel('Timestep (x1000)', fontsize=12)
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_title('Top 5 Polyhedra Over Time', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_atomic_volume(all_data: dict, output_path: Path):
    """原子体積分布をプロット"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 全体の体積分布
    ax = axes[0, 0]
    all_volumes = []
    for timestep, df in all_data.items():
        all_volumes.extend(df['Atomic_Volume'].values)

    ax.hist(all_volumes, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(all_volumes), color='red', linestyle='--',
               label=f'Mean: {np.mean(all_volumes):.2f} A³')
    ax.set_xlabel('Atomic Volume (A³)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Atomic Volume Distribution', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # タイプ別
    ax = axes[0, 1]
    type1_vol = []
    type2_vol = []
    for timestep, df in all_data.items():
        type1_vol.extend(df[df['Particle_Type'] == 1]['Atomic_Volume'].values)
        type2_vol.extend(df[df['Particle_Type'] == 2]['Atomic_Volume'].values)

    ax.hist(type1_vol, bins=40, alpha=0.6, label=f'Ga: {np.mean(type1_vol):.2f} A³',
            color='coral', edgecolor='black')
    ax.hist(type2_vol, bins=40, alpha=0.6, label=f'In: {np.mean(type2_vol):.2f} A³',
            color='skyblue', edgecolor='black')
    ax.set_xlabel('Atomic Volume (A³)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Atomic Volume by Type', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # 時間変化
    ax = axes[1, 0]
    timesteps = sorted(all_data.keys())
    mean_vols = [all_data[t]['Atomic_Volume'].mean() for t in timesteps]
    std_vols = [all_data[t]['Atomic_Volume'].std() for t in timesteps]

    ax.errorbar(timesteps, mean_vols, yerr=std_vols, marker='o',
                capsize=3, color='forestgreen', linewidth=2, markersize=6)
    ax.set_xlabel('Timestep (x1000)', fontsize=12)
    ax.set_ylabel('Mean Atomic Volume (A³)', fontsize=12)
    ax.set_title('Atomic Volume vs Time', fontsize=14)
    ax.grid(alpha=0.3)

    # 配位数vs体積
    ax = axes[1, 1]
    coords_all = []
    vols_all = []
    for timestep, df in all_data.items():
        coords_all.extend(df['Coordination'].values)
        vols_all.extend(df['Atomic_Volume'].values)

    ax.scatter(coords_all, vols_all, alpha=0.1, s=5, c='navy')

    # 平均線
    unique_coords = np.unique(coords_all)
    mean_vols_by_coord = [np.mean([v for c, v in zip(coords_all, vols_all) if c == uc])
                         for uc in unique_coords]
    ax.plot(unique_coords, mean_vols_by_coord, 'r-', linewidth=2, marker='o',
            markersize=8, label='Mean')

    ax.set_xlabel('Coordination Number', fontsize=12)
    ax.set_ylabel('Atomic Volume (A³)', fontsize=12)
    ax.set_title('Coordination vs Atomic Volume', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary(all_data: dict, output_path: Path):
    """サマリーグラフ"""

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. 配位数分布
    ax = axes[0, 0]
    all_coords = []
    for df in all_data.values():
        all_coords.extend(df['Coordination'].values)
    coords, counts = np.unique(all_coords, return_counts=True)
    ax.bar(coords, counts / len(all_coords) * 100, color='steelblue', edgecolor='black')
    ax.set_xlabel('Coordination Number')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Coordination Distribution')
    ax.grid(axis='y', alpha=0.3)

    # 2. 体積分布
    ax = axes[0, 1]
    all_vols = []
    for df in all_data.values():
        all_vols.extend(df['Atomic_Volume'].values)
    ax.hist(all_vols, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(all_vols), color='red', linestyle='--')
    ax.set_xlabel('Atomic Volume (A³)')
    ax.set_ylabel('Count')
    ax.set_title(f'Volume (Mean: {np.mean(all_vols):.1f} A³)')
    ax.grid(alpha=0.3)

    # 3. 総面数分布
    ax = axes[0, 2]
    total_faces = []
    for df in all_data.values():
        voronoi_cols = [c for c in df.columns if 'Voronoi_Index' in c]
        total_faces.extend(df[voronoi_cols].sum(axis=1).values)
    faces, counts = np.unique(total_faces, return_counts=True)
    ax.bar(faces, counts / len(total_faces) * 100, color='purple', edgecolor='black')
    ax.set_xlabel('Total Faces')
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Voronoi Faces Distribution')
    ax.grid(axis='y', alpha=0.3)

    # 4. 上位多面体
    ax = axes[1, 0]
    total_counter = Counter()
    for df in all_data.values():
        total_counter.update(analyze_voronoi_polyhedra(df))
    top10 = total_counter.most_common(10)
    labels = []
    for p in top10:
        idx = list(p[0])
        while idx and idx[-1] == 0:
            idx.pop()
        labels.append(f"<{','.join(map(str, idx))}>")
    percentages = [p[1] / sum(total_counter.values()) * 100 for p in top10]
    ax.barh(range(len(labels)), percentages, color='teal', edgecolor='black')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Frequency (%)')
    ax.set_title('Top 10 Voronoi Polyhedra')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 5. 5角形vs6角形
    ax = axes[1, 1]
    penta = []
    hexa = []
    for df in all_data.values():
        penta.extend(df['Voronoi_Index_5'].values)
        hexa.extend(df['Voronoi_Index_6'].values)
    ax.hist2d(penta, hexa, bins=[range(0, 15), range(0, 12)], cmap='YlGnBu', cmin=1)
    ax.set_xlabel('Pentagonal Faces')
    ax.set_ylabel('Hexagonal Faces')
    ax.set_title('Face Distribution')
    plt.colorbar(ax.collections[0], ax=ax)

    # 6. 統計サマリー
    ax = axes[1, 2]
    ax.axis('off')

    n_atoms = len(list(all_data.values())[0])
    n_timesteps = len(all_data)
    mean_coord = np.mean(all_coords)
    mean_vol = np.mean(all_vols)
    mean_faces = np.mean(total_faces)

    type1_count = sum(1 for df in all_data.values() for t in df['Particle_Type'] if t == 1)
    type2_count = sum(1 for df in all_data.values() for t in df['Particle_Type'] if t == 2)

    summary_text = f"""
    ═══════════════════════════════════
           Voronoi Analysis Summary
    ═══════════════════════════════════

    Number of atoms:     {n_atoms}
    Number of frames:    {n_timesteps}

    Atom types:
      Ga (Type 1):       {type1_count // n_timesteps}
      In (Type 2):       {type2_count // n_timesteps}

    ───────────────────────────────────

    Mean Coordination:   {mean_coord:.2f}
    Mean Atomic Volume:  {mean_vol:.2f} A³
    Mean Total Faces:    {mean_faces:.2f}

    Most common polyhedra:
      1. {labels[0]}: {percentages[0]:.1f}%
      2. {labels[1]}: {percentages[1]:.1f}%
      3. {labels[2]}: {percentages[2]:.1f}%

    ═══════════════════════════════════
    """

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """メイン処理"""

    print("=" * 60)
    print("Voronoi Analysis Visualization")
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
        print(f"  Reading {f.name}...", end=" ")
        df = read_voronoi_file(f)
        all_data[timestep] = df
        print(f"({len(df)} atoms)")

    print("\nGenerating plots...")

    # グラフ作成
    plot_coordination_distribution(all_data, OUTPUT_DIR / "voronoi_coordination.png")
    plot_voronoi_polyhedra(all_data, OUTPUT_DIR / "voronoi_polyhedra.png")
    plot_atomic_volume(all_data, OUTPUT_DIR / "voronoi_volume.png")
    plot_summary(all_data, OUTPUT_DIR / "voronoi_summary.png")

    print("\n" + "=" * 60)
    print("Done! Output files:")
    for f in OUTPUT_DIR.glob("voronoi_*.png"):
        print(f"  {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
