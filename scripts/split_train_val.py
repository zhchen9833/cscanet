import argparse
import os
import shutil
import random

from sklearn.model_selection import train_test_split


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Splitting dataset')
    parser.add_argument('--seed', type=int, default=49)
    parser.add_argument('--root', type=str, default='/home/disk2/ZR/datasets/OurHSI/128/npy/gt')
    parser.add_argument('--target', type=str, default='/home/disk2/ZR/datasets/OurHSI/128/npy')
    parser.add_argument("--train", type=str, default='data/npy/train/reflectivity')
    parser.add_argument("--val", type=str, default='data/npy/val/reflectivity')
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--nproc", type=int, default=10)
    _args = parser.parse_args()
    return _args


def move_split():
    _args = parse_args()
    # random seed
    random.seed(_args.seed)

    # source destination
    train_dir = _args.train
    val_dir = _args.val
    os.makedirs(val_dir, exist_ok=True)

    # get data
    all_images = os.listdir(train_dir)
    all_images = sorted(all_images, key=lambda x: int(x.split('.')[0]))

    # split ratio
    val_ratio = _args.ratio

    # random split
    val_size = int(len(all_images) * val_ratio)
    val_images = random.sample(all_images, val_size)

    # move
    for image in val_images:
        src_path = os.path.join(train_dir, image)
        dst_path = os.path.join(val_dir, image)
        shutil.move(str(src_path), str(dst_path))
        print(f'Move {src_path} data to {dst_path}')


def index_split(root_dir, target_dir, train_ratio=0.8, random_seed=42):
    # 1. 获取所有npy文件并排序(确保可重复性)
    all_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.npy')])
    if not all_files:
        raise ValueError(f"目录 {root_dir} 中未找到.npy文件")

    # 2. 划分训练集和验证集
    train_files, val_files = train_test_split(
        all_files,
        train_size=train_ratio,
        random_state=random_seed
    )

    # 3. 创建输出目录(如果不存在)
    os.makedirs(target_dir, exist_ok=True)

    # 4. 保存索引文件
    def save_index(files, filename):
        with open(os.path.join(target_dir, filename), 'w') as f:
            f.write("\n".join(files))

    save_index(train_files, "train.txt")
    save_index(val_files, "val.txt")

    print(f"划分完成: 共 {len(all_files)} 个文件")
    print(f"训练集: {len(train_files)} 个文件 (保存到 splits/train_files.txt)")
    print(f"验证集: {len(val_files)} 个文件 (保存到 splits/val_files.txt)")


if __name__ == '__main__':
    args = parse_args()
    index_split(
        root_dir=args.root,
        target_dir=args.target,
        train_ratio=0.9,
        random_seed=args.seed
    )
