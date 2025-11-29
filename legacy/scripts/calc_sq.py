#!/usr/bin/env python3
import argparse
import sys
import MDAnalysis as mda
import freud
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    # --- 引数の設定 ---
    parser = argparse.ArgumentParser(description='Calculate S(Q) from LAMMPS trajectory.')
    
    # 必須の引数: 入力ファイル
    parser.add_argument('input_file', type=str, help='Input LAMMPS trajectory file (e.g., dump.lammpstrj)')
    
    # オプション引数
    parser.add_argument('-o', '--output', type=str, default='sq_plot.png', help='Output plot filename (default: sq_plot.png)')
    parser.add_argument('--data', type=str, default=None, help='Output text data filename (optional)')
    parser.add_argument('--kmax', type=float, default=20.0, help='Maximum k vector magnitude (default: 20.0)')
    parser.add_argument('--kmin', type=float, default=0.5, help='Minimum k vector magnitude (default: 0.5)')
    parser.add_argument('--bins', type=int, default=100, help='Number of k bins (default: 100)')
    parser.add_argument('--step', type=int, default=1, help='Trajectory stride (step size) to speed up calc (default: 1)')

    args = parser.parse_args()

    # --- ファイルの読み込み ---
    print(f"Loading trajectory: {args.input_file} ...")
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)

    try:
        u = mda.Universe(args.input_file, format='LAMMPSDUMP')
    except Exception as e:
        print(f"Error loading MDAnalysis universe: {e}")
        sys.exit(1)

    print(f"  - Atoms: {len(u.atoms)}")
    print(f"  - Frames: {len(u.trajectory)}")

    # --- S(k) 計算機の準備 ---
    sf = freud.diffraction.StaticStructureFactorDirect(
        bins=args.bins,
        k_max=args.kmax,
        k_min=args.kmin
    )

    # --- 計算ループ (平均化処理) ---
    print(f"Calculating S(k) (Step: {args.step})...")

    frame_count = 0

    # フレームごとに計算してfreudに累積させる
    for ts in u.trajectory[::args.step]:
        # freudのボックス形式に変換
        box = freud.box.Box.from_box(u.dimensions)
        positions = u.atoms.positions

        # freudで計算 (reset=Falseで累積)
        if frame_count == 0:
            sf.compute((box, positions), reset=True)
        else:
            sf.compute((box, positions), reset=False)

        frame_count += 1

        # 進捗表示
        if frame_count % 10 == 0:
            print(f"\r  Processed {frame_count} frames...", end="")

    print(f"\nCalculation finished. Averaged over {frame_count} frames.")

    # 結果の取得 (freudが自動的に平均化)
    S_k_avg = sf.S_k
    bin_centers = sf.bin_centers

    # --- データの保存 (オプション) ---
    if args.data:
        print(f"Saving raw data to {args.data} ...")
        header = "k, S(k)"
        np.savetxt(args.data, np.column_stack((bin_centers, S_k_avg)), header=header, delimiter=',')

    # --- プロット ---
    print(f"Saving plot to {args.output} ...")
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, S_k_avg, label=r'Averaged $S(k)$', color='b')
    plt.xlabel(r'$k \ (\mathrm{\AA}^{-1})$')
    plt.ylabel(r'$S(k)$')
    plt.title(f'Static Structure Factor\nSrc: {args.input_file}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')

    plt.savefig(args.output, dpi=300)
    print("Done.")

if __name__ == "__main__":
    main()
