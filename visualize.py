import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_rul_comparison(y_test, results,
                        save_path='outputs/rul_comparison.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (name, result) in enumerate(results.items()):
        axes[idx].scatter(
            y_test,
            result['predictions'],
            alpha=0.4,
            s=10,
            color='steelblue'
        )
        max_val = max(y_test.max(), result['predictions'].max())
        axes[idx].plot(
            [0, max_val], [0, max_val],
            'r--', linewidth=1.5,
            label='Perfect prediction'
        )
        axes[idx].set_xlabel('Actual RUL (cycles)')
        axes[idx].set_ylabel('Predicted RUL (cycles)')
        axes[idx].set_title(
            f'{name}\nRMSE: {result["rmse"]:.2f}  |  NASA Score: {result["nasa_score"]:.2f}'
        )
        axes[idx].legend()

    plt.suptitle('RUL Prediction — Actual vs Predicted', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_degradation_curve(train, unit_id=1,
                           save_path='outputs/degradation_curve.png'):
    engine = train[train['unit_id'] == unit_id].copy()
    actual_rul = engine['RUL'].values
    cycles = engine['cycle'].values

    plt.figure(figsize=(10, 5))
    plt.plot(
        cycles, actual_rul,
        label='Actual RUL',
        color='black',
        linewidth=2
    )
    plt.axhline(
        y=0, color='red', linestyle='--',
        linewidth=1, label='Failure point'
    )
    plt.xlabel('Engine Cycle')
    plt.ylabel('Remaining Useful Life (cycles)')
    plt.title(f'Engine {unit_id} — Degradation Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_rmse_comparison(results,
                         save_path='outputs/rmse_comparison.png'):
    names = list(results.keys())
    rmses = [r['rmse'] for r in results.values()]
    nasa_scores = [r['nasa_score'] for r in results.values()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bars1 = axes[0].bar(
        names, rmses,
        color=['steelblue', 'darkorange'],
        width=0.4
    )
    axes[0].set_ylabel('RMSE (cycles)')
    axes[0].set_title('Model Comparison — RMSE\n(lower is better)')
    for bar, val in zip(bars1, rmses):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{val:.2f}',
            ha='center', fontsize=11
        )

    bars2 = axes[1].bar(
        names, nasa_scores,
        color=['steelblue', 'darkorange'],
        width=0.4
    )
    axes[1].set_ylabel('NASA Score')
    axes[1].set_title('Model Comparison — NASA Score\n(lower is better, penalizes late predictions)')
    for bar, val in zip(bars2, nasa_scores):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{val:.2f}',
            ha='center', fontsize=11
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")