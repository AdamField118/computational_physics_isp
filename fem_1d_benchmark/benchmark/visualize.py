#!/usr/bin/env python3
"""Write static timing plots and a Markdown table from the saved FEM benchmark."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / 'results'


def generate(input_path, output_dir):
    data = json.loads(Path(input_path).read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benches = data['benchmarks']
    baseline = {r['n']: r['mean'] for b in benches if b['name'] == 'Python' for r in b['results']}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    lines = ['# FEM assembly timings', '',
             'Plots are generated locally; see the [figure instructions](../../docs/GENERATED_FILES.md).', '',
             'Saved run: ' + data['metadata']['date'], '',
             'Times are in milliseconds. Speedup is relative to the Python wrapper at the same n.', '',
             '| Implementation | n | Mean | Std. dev. | Minimum | Maximum | Speedup |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for b in benches:
        rows = b['results']
        n = [r['n'] for r in rows]
        means = [r['mean'] * 1000 for r in rows]
        ratios = [baseline[r['n']] / r['mean'] for r in rows]
        axes[0].loglog(n, means, 'o-', label=b['name'])
        axes[1].semilogx(n, ratios, 'o-', label=b['name'])
        for r, speed in zip(rows, ratios):
            cells = [b['name'], str(r['n'])] + [f"{r[k]*1000:.6f}" for k in ['mean', 'std', 'min', 'max']] + [f'{speed:.3f}']
            lines.append('| ' + ' | '.join(cells) + ' |')
    axes[0].set(xlabel='Elements n', ylabel='Assembly time (ms)')
    axes[1].set(xlabel='Elements n', ylabel='Speedup over Python')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output_dir / 'assembly_comparison.png', dpi=170)
    plt.close(fig)
    lines += ['', 'Figure: Assembly time and speedup. Generated locally as `assembly_comparison.png`.', '']
    (output_dir / 'benchmark_summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path, default=RESULTS / 'fem_benchmark_results.json')
    parser.add_argument('--output-dir', type=Path, default=RESULTS)
    args = parser.parse_args()
    generate(args.input, args.output_dir)
    print(f'Saved figures and Markdown table to {args.output_dir}')


if __name__ == '__main__':
    main()
