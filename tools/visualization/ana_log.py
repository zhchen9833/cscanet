import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

# === 配色定义（科研常用）===
COLORS = {
    'loss': '#1f77b4',
    'lr': '#ff7f0e',
    'psnr': '#2ca02c',
    'ssim': '#d62728',
    'best_marker': 'gold',
    'best_edge': 'black'
}


def parse_log(log_path):
    train_data = []
    val_data = []
    pending_psnr = None
    pending_ssim = None

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            pattern = re.compile(
                r'epoch:\[(\d+)/(\d+)]\s*iter:\[(\d+)/(\d+)]\s*loss:\s*([\d.]+)\s*lr:([\d.e+-]+)'
            )
            # 匹配训练记录：epoch:[84/100]	iter:[50/2520]	loss: ...	lr: ...
            train_match = re.search(
                pattern,
                line)
            if train_match:
                epoch = int(train_match.group(1))
                cur_iter = int(train_match.group(3))
                total_iter = int(train_match.group(4))
                loss = float(train_match.group(5))
                lr = float(train_match.group(6))
                train_data.append({
                    'epoch': epoch,
                    'iter': cur_iter + (epoch-1) * total_iter,
                    'loss': loss,
                    'lr': lr
                })

            # 匹配验证记录："Mean PSNR: XX.XX dB"
            if "Mean PSNR" in line:
                pending_psnr = re.search(r'PSNR:\s*([\d.]+)', line).group(1)
            if "Mean SSIM" in line:
                pending_ssim = re.search(r'SSIM:\s*([\d.]+)', line).group(1)

            if pending_psnr and pending_ssim:
                psnr = float(pending_psnr)
                ssim = float(pending_ssim)
                val_data.append({
                    'epoch': epoch,
                    'psnr': psnr,
                    'ssim': ssim
                })
                pending_psnr = None
                pending_ssim = None

    # 转换为 DataFrame
    _train_df = pd.DataFrame(train_data)
    _val_df = pd.DataFrame(val_data)

    return _train_df, _val_df


def plot_single_experiment(_train_df, _val_df, exp_name="Experiment"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss vs Iter
    axes[0, 0].plot(_train_df['iter'], _train_df['loss'], label='Loss', color=COLORS['loss'])
    axes[0, 0].set_title(f'{exp_name} - Training Loss')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # LR vs Epoch
    axes[0, 1].plot(_train_df['epoch'], _train_df['lr'], label='Learning Rate', color=COLORS['lr'])
    axes[0, 1].set_title(f'{exp_name} - Learning Rate')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('LR')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # 设置 y 轴为科学计数法格式
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))  # 小于 1e-3 或大于 1e3 才用科学记数法
    axes[0, 1].yaxis.set_major_formatter(formatter)
    axes[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # 强制科学记数法

    # PSNR vs Epoch
    axes[1, 0].plot(_val_df['epoch'], _val_df['psnr'], label='PSNR', color=COLORS['psnr'])
    best_psnr_row = _val_df.loc[_val_df['psnr'].idxmax()]
    axes[1, 0].scatter(best_psnr_row['epoch'], best_psnr_row['psnr'],
                       color=COLORS['best_marker'], edgecolor=COLORS['best_edge'], s=100, zorder=5,
                       label=f"Best: {best_psnr_row['psnr']:.2f} dB")
    axes[1, 0].set_title(f'{exp_name} - Validation PSNR')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # SSIM vs Epoch
    axes[1, 1].plot(_val_df['epoch'], _val_df['ssim'], label='SSIM', color=COLORS['ssim'])
    best_ssim_row = _val_df.loc[_val_df['ssim'].idxmax()]
    axes[1, 1].scatter(best_ssim_row['epoch'], best_ssim_row['ssim'],
                       color=COLORS['best_marker'], edgecolor=COLORS['best_edge'], s=100, zorder=5,
                       label=f"Best: {best_ssim_row['ssim']:.4f}")
    axes[1, 1].set_title(f'{exp_name} - Validation SSIM')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('SSIM')
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{exp_name}_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()


def compare_experiments(experiment_paths, names=None, save_name="comparison_results.png"):
    if names is None:
        names = [f"Exp{i + 1}" for i in range(len(experiment_paths))]

    plt.figure(figsize=(12, 6))

    # PSNR 对比
    plt.subplot(1, 2, 1)
    for path, name in zip(experiment_paths, names):
        _, _val_df = parse_log(path)
        best_psnr_row = _val_df.loc[_val_df['psnr'].idxmax()]
        plt.plot(_val_df['epoch'], _val_df['psnr'], label=name, marker='o', linestyle='-', markersize=4,
                 color=COLORS['psnr'] if name == names[0] else COLORS['loss'])
        plt.scatter(best_psnr_row['epoch'], best_psnr_row['psnr'],
                    color=COLORS['best_marker'], edgecolor=COLORS['best_edge'], s=80, zorder=5)
    plt.title("PSNR Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # SSIM 对比
    plt.subplot(1, 2, 2)
    for path, name in zip(experiment_paths, names):
        _, _val_df = parse_log(path)
        best_ssim_row = _val_df.loc[_val_df['ssim'].idxmax()]
        plt.plot(_val_df['epoch'], _val_df['ssim'], label=name, marker='x', linestyle='--', markersize=4,
                 color=COLORS['ssim'] if name == names[0] else COLORS['lr'])
        plt.scatter(best_ssim_row['epoch'], best_ssim_row['ssim'],
                    color=COLORS['best_marker'], edgecolor=COLORS['best_edge'], s=80, zorder=5)
    plt.title("SSIM Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    train_df, val_df = parse_log(
        r"/home/disk1/ZR/PythonProjects/uchiha/experiments/hdr_former/AVIRIS/baseline_v2/logs/2025-05-30-15-47.log")
    plot_single_experiment(train_df, val_df, exp_name="baseline_v2")

    compare_experiments(
        experiment_paths=[
            r"/home/disk1/ZR/PythonProjects/uchiha/experiments/hdr_former/AVIRIS/baseline_v2/logs/2025-05-30-15-47.log",
            r"/home/disk1/ZR/PythonProjects/uchiha/experiments/hdr_former/AVIRIS/baseline/logs/2025-05-22-10-57.log"
        ],
        names=["baseline_v2", "baseline_v1"]
    )
