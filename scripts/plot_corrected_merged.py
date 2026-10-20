"""Plot saved balanced training metrics with their distinct denominators."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='outputs/merged-corrected-20261001')
    args = parser.parse_args()
    run = ROOT / args.run
    summary = json.loads((run / 'training_summary.json').read_text(encoding='utf-8'))
    with (run / 'training.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == summary['epochs_run']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [int(row['epoch']) + 1 for row in rows]
    chosen = summary['best_epoch']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    axes[0].plot(epochs, [float(r['loss']) for r in rows], label='Train: sampled + augmentation')
    axes[0].plot(epochs, [float(r['val_loss']) for r in rows], label='Validation: raw frames')
    axes[0].set_title('Cross entropy (different distributions)')
    axes[0].set_ylabel('Loss')
    axes[1].plot(epochs, [float(r['val_source_class_video_loss']) for r in rows],
                 color='tab:green', label='Mean by video, class, source')
    axes[1].set_title('Validation selection metric')
    axes[1].set_ylabel('Cross entropy')
    axes[2].plot(epochs, [float(r['val_macro_f1']) * 100 for r in rows], label='Validation macro F1')
    axes[2].set_title('Validation original frames')
    axes[2].set_ylabel('Macro F1 (%)')
    for axis in axes:
        axis.axvline(chosen, linestyle='--', color='0.35', label=f'Selected epoch {chosen}')
        axis.set_xlabel('Balanced epoch (warmup excluded)')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('Corrected merged training: 1 full warmup pass, then balanced sampling')
    fig.savefig(run / 'learning_curves.png', dpi=150)
    plt.close(fig)
    print(run / 'learning_curves.png')


if __name__ == '__main__':
    main()
