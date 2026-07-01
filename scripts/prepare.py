import h5py
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


def crop_and_save_npy(src_path, dst_path, crop_size=128):
    """
    裁剪NPY文件并保存到目标路径（支持H×W×C三维数据）

    参数:
        src_path: 原始数据根目录 (包含 gt/ 和 rain/)
        dst_path: 目标保存根目录
        crop_size: 裁剪块大小 (默认128x128)
    """
    # 定义所有需要处理的子目录
    sub_dirs = ['gt'] + [f'rain/{t}' for t in ['small', 'medium', 'heavy', 'storm']]

    # 主进度条（目录级）
    with tqdm(sub_dirs, desc="处理目录", position=0) as pbar_dirs:
        for subdir in pbar_dirs:
            src_dir = os.path.join(src_path, subdir)
            dst_dir = os.path.join(dst_path, subdir)
            os.makedirs(dst_dir, exist_ok=True)

            # 获取当前目录下所有NPY文件
            npy_files = list(Path(src_dir).glob('*.npy'))

            # 文件级进度条
            with tqdm(npy_files, desc=f"处理 {subdir}", position=1, leave=False) as pbar_files:
                for npy_file in pbar_files:
                    try:
                        data = np.load(npy_file)
                        h, w, c = data.shape

                        # 动态更新进度条描述
                        pbar_files.set_postfix_str(f"正在处理 {npy_file.name}")

                        # 检查尺寸
                        if h % crop_size != 0 or w % crop_size != 0:
                            tqdm.write(f"⚠️ 跳过: {npy_file.name} (尺寸{h}x{w}x{c}不匹配)")
                            continue

                        # 计算裁剪块数
                        num_h = h // crop_size
                        num_w = w // crop_size

                        # 裁剪并保存
                        for i in range(num_h):
                            for j in range(num_w):
                                crop_data = data[
                                            i * crop_size: (i + 1) * crop_size,
                                            j * crop_size: (j + 1) * crop_size,
                                            :
                                            ]
                                save_name = f"{npy_file.stem}_x{i * crop_size}_y{j * crop_size}.npy"
                                np.save(os.path.join(dst_dir, save_name), crop_data)

                        # 使用tqdm.write保证输出对齐
                        tqdm.write(f"✅ 完成: {npy_file.name} → 生成 {num_h * num_w}个块")

                    except Exception as e:
                        tqdm.write(f"❌ 错误: {npy_file.name} - {str(e)}")
                        continue


def convert_to_hdf5(src_path, h5_path):
    """
    将裁剪后的NPY样本对转为HDF5格式

    参数:
        src_path: 包含gt和rain子目录的根路径
        h5_path: 输出的HDF5文件路径
    """
    # 初始化HDF5文件（启用压缩和SWMR模式）
    with h5py.File(h5_path, 'w') as hf:
        # 获取所有样本名（假设gt和rain的各子目录文件名完全一致）
        sample_names = [f.stem for f in Path(src_path, 'gt').glob('*.npy')]

        # 创建组结构
        gt_group = hf.create_group('gt')
        rain_group = hf.create_group('rain')
        rain_subgroups = {
            'small': rain_group.create_group('small'),
            'medium': rain_group.create_group('medium'),
            'heavy': rain_group.create_group('heavy'),
            'storm': rain_group.create_group('storm')
        }

        # 遍历所有样本
        for name in tqdm(sample_names, desc='转换样本'):
            # 加载GT数据
            gt_data = np.load(Path(src_path, 'gt', f'{name}.npy'))
            gt_group.create_dataset(name, data=gt_data, compression='lzf')

            # 加载各噪声类型数据
            for noise_type in ['small', 'medium', 'heavy', 'storm']:
                rain_data = np.load(Path(src_path, 'rain', noise_type, f'{name}.npy'))
                rain_subgroups[noise_type].create_dataset(name, data=rain_data, compression='lzf')


def create_h5_for_debug(src_path, h5_path, train_list_path, val_list_path):
    """
    创建统一的HDF5文件，包含所有样本并标记训练/验证集

    参数:
        src_path: 包含gt和rain子目录的根路径
        h5_path: 输出的HDF5文件路径
        train_list_path: 训练集文件列表的文本文件路径
        val_list_path: 验证集文件列表的文本文件路径
    """

    # 读取训练集和验证集文件列表
    def read_file_list(path):
        with open(path, 'r') as f:
            return [line.strip().replace('.npy', '') for line in f if line.strip()]

    train_names = set(read_file_list(train_list_path))
    val_names = set(read_file_list(val_list_path))

    # 检查是否有重叠文件
    overlap = train_names & val_names
    if overlap:
        print(f"警告: {len(overlap)}个文件同时出现在训练集和验证集中")

    all_names = train_names | val_names
    print(f"总样本数: {len(all_names)} (训练: {len(train_names)}, 验证: {len(val_names)})")

    # 创建HDF5文件
    with h5py.File(h5_path, 'w') as hf:
        # 创建主组结构
        gt_group = hf.create_group('gt')
        rain_group = hf.create_group('rain')
        rain_subgroups = {
            'small': rain_group.create_group('small'),
            'medium': rain_group.create_group('medium'),
            'heavy': rain_group.create_group('heavy'),
            'storm': rain_group.create_group('storm')
        }

        # 添加文件集标记属性
        hf.attrs['train_files'] = list(train_names)
        hf.attrs['val_files'] = list(val_names)

        # 处理所有文件
        for name in tqdm(all_names, desc='构建HDF5数据集'):
            # 标记样本类型属性
            sample_type = 'train' if name in train_names else 'val'

            # 处理GT数据
            gt_data = np.load(Path(src_path, 'gt', f'{name}.npy'))
            gt_dset = gt_group.create_dataset(name, data=gt_data, compression='lzf')
            gt_dset.attrs['split'] = sample_type

            # 处理各降雨类型数据
            for rain_type, subgroup in rain_subgroups.items():
                rain_data = np.load(Path(src_path, 'rain', rain_type, f'{name}.npy'))
                rain_dset = subgroup.create_dataset(name, data=rain_data, compression='lzf')
                rain_dset.attrs['split'] = sample_type


if __name__ == '__main__':
    # 使用示例
    # crop_and_save_npy(
    #     src_path="/home/disk2/ZR/datasets/OurHSI/256",  # 替换为原始数据路径
    #     dst_path="/home/disk2/ZR/datasets/OurHSI/128/npy",  # 替换为目标保存路径
    #     crop_size=128
    # )
    #
    # convert_to_hdf5(
    #     src_path="/home/disk2/ZR/datasets/OurHSI/128/npy",  # 裁剪后的NPY块路径
    #     h5_path="/home/disk2/ZR/datasets/OurHSI/128/npy/dataset.h5"  # 输出的HDF5文件路径
    # )

    create_h5_for_debug(
        src_path='/home/disk2/ZR/datasets/AVIRIS/128/npy',
        h5_path='/home/disk2/ZR/datasets/AVIRIS/128/npy/debug.h5',
        train_list_path='/home/disk2/ZR/datasets/AVIRIS/128/npy/train_for_debug.txt',
        val_list_path='/home/disk2/ZR/datasets/AVIRIS/128/npy/val_for_debug.txt'
    )
