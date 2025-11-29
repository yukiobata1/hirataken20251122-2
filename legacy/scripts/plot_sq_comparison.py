#!/usr/bin/env python3
"""
Plot S(q) comparison for multiple temperatures
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_sq_data(filepath):
    """Load S(q) data from file"""
    data = np.loadtxt(filepath)
    q = data[:, 0]
    sq = data[:, 1]
    return q, sq

def main():
    parser = argparse.ArgumentParser(description='Plot S(q) comparison for multiple temperatures')
    parser.add_argument('--output', '-o', default='sq_comparison.png', help='Output plot filename')
    args = parser.parse_args()

    # Temperature data files
    temps = ['293K', '600K', '1000K']
    colors = ['blue', 'orange', 'red']

    plt.figure(figsize=(10, 6))

    for temp, color in zip(temps, colors):
        filepath = f'sq_{temp}.dat'
        try:
            q, sq = load_sq_data(filepath)
            # Filter for q >= 1.0
            mask = q >= 1.0
            plt.plot(q[mask], sq[mask], label=f'{temp}', color=color, linewidth=2)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found, skipping...")

    plt.xlabel('q (Å⁻¹)', fontsize=14)
    plt.ylabel('S(q)', fontsize=14)
    plt.title('Structure Factor Comparison', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {args.output}")

if __name__ == '__main__':
    main()
