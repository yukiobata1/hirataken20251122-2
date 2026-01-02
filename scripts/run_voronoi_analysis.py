#!/usr/bin/env python3
"""
Voronoi解析用のシミュレーションを実行し、トラジェクトリファイルを出力する。
OVITO用に .lammpstrj, .xyz, .pdb 形式で保存。
"""

import subprocess
import os
import numpy as np
from pathlib import Path

# 設定
OUTPUT_DIR = Path("outputs/fine_search_shoulder/voronoi")
DATA_FILE = "inputs/data.ga_base_2types"

# ベストフィットパラメータ (metrics_summary.csv より)
BEST_PARAMS = {
    "sig12": 1.17,
    "ga1_frac": 0.45,
}

# LJポテンシャルパラメータ
BASE_SIGMA_GA1 = 2.70  # Ga1-Ga1 (大きい原子)
BASE_SIGMA_GA2 = 2.70  # Ga2-Ga2 (小さい原子)
EPSILON = 0.43  # kcal/mol
SIGMA_GA1_FACTOR = 1.1
SIGMA_GA2_FACTOR = 0.9

def create_input_file(output_dir: Path, params: dict) -> Path:
    """Voronoi解析用のLAMMPS入力ファイルを作成"""

    sig12 = params["sig12"]
    ga1_frac = params["ga1_frac"]

    # sigma値の計算
    sigma_ga1 = BASE_SIGMA_GA1 * SIGMA_GA1_FACTOR
    sigma_ga2 = BASE_SIGMA_GA2 * SIGMA_GA2_FACTOR
    sigma_ga12 = BASE_SIGMA_GA1 * sig12

    input_content = f"""# Voronoi Analysis Run
# Best fit: Ga1={ga1_frac}, sigma12={sig12}x
# Ga1-Ga1 sigma = {sigma_ga1:.4f} A (x{SIGMA_GA1_FACTOR})
# Ga2-Ga2 sigma = {sigma_ga2:.4f} A (x{SIGMA_GA2_FACTOR})
# Ga1-Ga2 sigma = {sigma_ga12:.4f} A (x{sig12})
# Epsilon = {EPSILON:.4f} kcal/mol

package         kokkos neigh full newton off

units           real
atom_style      atomic
boundary        p p p

read_data       {DATA_FILE}

set             group all type 1
set             group all type/fraction 2 {ga1_frac} 18495

pair_style      lj/cut/kk 12.0
pair_coeff      1 1 {EPSILON:.4f} {sigma_ga1:.4f}
pair_coeff      2 2 {EPSILON:.4f} {sigma_ga2:.4f}
pair_coeff      1 2 {EPSILON:.4f} {sigma_ga12:.4f}

neighbor        2.0 bin
neigh_modify    delay 0 every 1 check yes
velocity        all create 423.15 18495 dist gaussian

thermo          1000
thermo_style    custom step temp press density

minimize        1.0e-4 1.0e-6 10000 100000
reset_timestep  0

# 平衡化
timestep        2.0
fix             nvt all nvt temp 423.15 423.15 100.0
run             50000
unfix           nvt
reset_timestep  0

# 本番: トラジェクトリ出力
fix             nvt all nvt temp 423.15 423.15 100.0

# LAMMPS dump形式 (.lammpstrj) - 全原子情報
dump            traj all custom 1000 {output_dir}/trajectory.lammpstrj id type x y z vx vy vz

# XYZ形式
dump            xyz all xyz 1000 {output_dir}/trajectory.xyz
dump_modify     xyz element Ga In

# RDF計算
compute         myrdf all rdf 200 1 1 2 2 1 2 cutoff 12.0
fix             rdfout all ave/time 100 1000 100000 c_myrdf[*] file {output_dir}/output.rdf mode vector

run             100000

# 最終構造を出力
write_data      {output_dir}/final_structure.data
write_dump      all custom {output_dir}/final_structure.lammpstrj id type x y z vx vy vz
"""

    input_file = output_dir / "in.voronoi"
    input_file.write_text(input_content)
    return input_file


def convert_lammpstrj_to_pdb(lammpstrj_file: Path, pdb_file: Path, frame: int = -1):
    """LAMMPSトラジェクトリをPDB形式に変換（最終フレームまたは指定フレーム）"""

    print(f"Converting {lammpstrj_file} to PDB...")

    frames = []
    current_frame = None
    box_bounds = None

    with open(lammpstrj_file, 'r') as f:
        for line in f:
            if "ITEM: TIMESTEP" in line:
                if current_frame is not None:
                    frames.append((current_frame, box_bounds))
                current_frame = []
                box_bounds = []
            elif "ITEM: BOX BOUNDS" in line:
                # 次の3行がボックス境界
                for _ in range(3):
                    bounds = next(f).strip().split()
                    box_bounds.append([float(bounds[0]), float(bounds[1])])
            elif "ITEM: ATOMS" in line:
                # ヘッダー行を読み飛ばし
                pass
            elif current_frame is not None and not line.startswith("ITEM:"):
                parts = line.strip().split()
                if len(parts) >= 4:
                    atom_id = int(parts[0])
                    atom_type = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    current_frame.append((atom_id, atom_type, x, y, z))

        if current_frame is not None:
            frames.append((current_frame, box_bounds))

    if not frames:
        print("No frames found in trajectory!")
        return

    # 指定フレームを取得（-1は最終フレーム）
    atoms, box = frames[frame]

    # PDBファイルを書き出し
    with open(pdb_file, 'w') as f:
        f.write("HEADER    LAMMPS STRUCTURE\n")
        f.write("TITLE     Voronoi Analysis Structure\n")

        # CRYST1レコード（ボックスサイズ）
        if box:
            lx = box[0][1] - box[0][0]
            ly = box[1][1] - box[1][0]
            lz = box[2][1] - box[2][0]
            f.write(f"CRYST1{lx:9.3f}{ly:9.3f}{lz:9.3f}  90.00  90.00  90.00 P 1           1\n")

        for atom_id, atom_type, x, y, z in atoms:
            # Type 1 = Ga, Type 2 = In
            element = "GA" if atom_type == 1 else "IN"
            atom_name = element.ljust(4)
            res_name = "LIQ"
            f.write(f"ATOM  {atom_id:5d} {atom_name}{res_name:>4} A{1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n")

        f.write("END\n")

    print(f"Saved PDB: {pdb_file}")


def run_simulation(input_file: Path, output_dir: Path):
    """LAMMPSシミュレーションを実行（GPU/Kokkos版）"""

    log_file = output_dir / "log.lammps"
    cmd = f"lmp -k on g 1 -sf kk -in {input_file} -log {log_file}"

    print(f"Running LAMMPS simulation (GPU/Kokkos)...")
    print(f"Input: {input_file}")
    print(f"Output directory: {output_dir}")

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=Path.cwd(),
            capture_output=True, text=True
        )
    except Exception as e:
        print(f"Error running simulation: {e}")
        return False

    if result.returncode == 0:
        print("Simulation completed successfully!")
        return True
    else:
        print(f"Simulation failed: {result.stderr}")
        return False


def main():
    """メイン処理"""

    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Voronoi Analysis Simulation")
    print("=" * 60)
    print(f"Best fit parameters:")
    print(f"  sigma12 = {BEST_PARAMS['sig12']}")
    print(f"  ga1_frac = {BEST_PARAMS['ga1_frac']}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # 入力ファイル作成
    input_file = create_input_file(OUTPUT_DIR, BEST_PARAMS)
    print(f"Created input file: {input_file}")

    # シミュレーション実行
    success = run_simulation(input_file, OUTPUT_DIR)

    if success:
        # LAMMPSトラジェクトリをPDBに変換
        lammpstrj_file = OUTPUT_DIR / "trajectory.lammpstrj"
        if lammpstrj_file.exists():
            convert_lammpstrj_to_pdb(
                lammpstrj_file,
                OUTPUT_DIR / "final_structure.pdb"
            )

        print("\n" + "=" * 60)
        print("Output files for OVITO:")
        print("=" * 60)
        for f in sorted(OUTPUT_DIR.glob("*")):
            print(f"  {f}")
        print("\nYou can open these files in OVITO:")
        print("  - trajectory.lammpstrj : Full trajectory")
        print("  - trajectory.xyz : XYZ format trajectory")
        print("  - final_structure.pdb : Final structure in PDB format")
        print("  - final_structure.lammpstrj : Final structure snapshot")
    else:
        print("Simulation failed. Check logs for details.")


if __name__ == "__main__":
    main()
