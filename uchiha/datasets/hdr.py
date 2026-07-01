import glob
import os
from typing import List, Dict

import h5py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tabulate import tabulate
from torch.utils.data import Dataset

from .builder import DATASET
from .pipelines import Compose
from ..utils import get_root_logger


@DATASET.register_module()
class HDF5MultiLevelRainHSIDataset(Dataset):
    """Multi-level noise HSI data set based on HDF5 storage

    Args:
        h5_path: HDF5 file path
        split_file: txt path of the divided file
        pipelines: data preprocessing pipeline
    """

    NOISE_LEVELS = ['small', 'medium', 'heavy', 'storm']

    def __init__(self, h5_path, split_file, pipelines=None):
        self.h5_path = h5_path
        self.pipelines = Compose(pipelines) if pipelines else None

        # load partition file
        with open(split_file, 'r') as f:
            self.sample_names = [line.strip() for line in f if line.strip()]

        # Initialize HDF5 connection (delayed until it is turned on for actual use)
        self._h5 = None
        self._validate_samples()

    @property
    def h5(self):
        """Delayed loading of HDF5 files (multi-process support)"""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        return self._h5

    def _validate_samples(self):
        """verify all samples are present in hdf5"""
        # no suffix in hdf5
        with h5py.File(self.h5_path, 'r') as hf:
            missing = []
            for name in self.sample_names:
                name = name.replace('.npy', '')
                # verify gt
                if f'gt/{name}' not in hf:
                    missing.append(f'gt/{name}')

                # verify each noise level
                for level in self.NOISE_LEVELS:
                    if f'rain/{level}/{name}' not in hf:
                        missing.append(f'rain/{level}/{name}')

            if missing:
                raise KeyError(f"Missing {len(missing)} samples in HDF5, e.g.: {missing[:3]}")

    def __len__(self):
        return len(self.sample_names) * len(self.NOISE_LEVELS)

    def __getitem__(self, idx):
        # Calculate the correspondence between sample name and noise level
        sample_idx = idx // len(self.NOISE_LEVELS)
        noise_idx = idx % len(self.NOISE_LEVELS)

        sample_name = self.sample_names[sample_idx]
        noise_level = self.NOISE_LEVELS[noise_idx]
        sample_name = sample_name.replace('.npy', '')

        # load data from hdf5 memory mapped
        gt = self.h5[f'gt/{sample_name}'][:]
        lq = self.h5[f'rain/{noise_level}/{sample_name}'][:]

        results = {
            'sample': lq,
            'target': gt,
            'index': idx,
            'noise_level': noise_level
        }

        return self.pipelines(results) if self.pipelines else results

    def evaluate(self, preds: List[np.ndarray], targets: List[np.ndarray], metric,
                 indexes: List[int]) -> dict:
        """
        Calculates the similarity between the predicted data and the real data. PSNR and SSIM are supported by default.

        Args:
            preds: Model output list, each element shape is (B, C, H, W)
            targets: A list of real labels, each element of shape (B, C, H, W)
            metric: metric to be calculated. Default value: ['PSNR','SSIM']
            indexes: raw data index list

        Returns:
            dict: dictionary with average psnr and ssim
        """
        assert len(preds) == len(targets) == len(indexes), "input list length must be the same"
        assert metric is None or metric.upper() == 'PSNR' or metric.upper() == 'SSIM', \
            'only psnr and ssim are supported by default'

        results = {
            'global': {'psnr': [], 'ssim': []},
            'by_level': {level: {'psnr': [], 'ssim': []} for level in self.NOISE_LEVELS}
        }
        logger = get_root_logger()
        logger.info('start evaluating...')

        for pred_batch, target_batch, index_batch in zip(preds, targets, indexes):
            B, C, H, W = pred_batch.shape

            for i in range(B):
                pred = pred_batch[i]  # (C,H,W)
                target = target_batch[i]
                idx = index_batch[i]

                # acquire noise level
                noise_level = self.NOISE_LEVELS[idx % len(self.NOISE_LEVELS)]

                # convert to (H W C) format
                if C == 1:
                    pred_img = pred[0]
                    target_img = target[0]
                else:
                    pred_img = np.transpose(pred, (1, 2, 0))
                    target_img = np.transpose(target, (1, 2, 0))

                # calculate psnr
                data_range = target_img.max() - target_img.min()
                psnr_val = psnr(target_img, pred_img, data_range=data_range)

                # calculate ssim
                multichannel = C > 1
                ssim_val = ssim(
                    target_img, pred_img,
                    multichannel=multichannel,
                    channel_axis=2 if multichannel else None,
                    data_range=data_range
                )

                # store results
                results['global']['psnr'].append(psnr_val)
                results['global']['ssim'].append(ssim_val)
                results['by_level'][noise_level]['psnr'].append(psnr_val)
                results['by_level'][noise_level]['ssim'].append(ssim_val)

        # computational statistics
        def calc_stats(values):
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }

        stats = {
            'global': {
                'psnr': calc_stats(results['global']['psnr']),
                'ssim': calc_stats(results['global']['ssim'])
            },
            'by_level': {
                level: {
                    'psnr': calc_stats(results['by_level'][level]['psnr']),
                    'ssim': calc_stats(results['by_level'][level]['ssim'])
                } for level in self.NOISE_LEVELS
            }
        }

        # log output
        logger.info(f"{' Evaluation Results ':=^60}")

        # Build hierarchical indicator tables
        table_data = []
        for level in self.NOISE_LEVELS:
            table_data.append([
                level.upper(),
                f"{stats['by_level'][level]['psnr']['mean']:.2f} ± {stats['by_level'][level]['psnr']['std']:.2f}",
                f"{stats['by_level'][level]['psnr']['min']:.2f}-{stats['by_level'][level]['psnr']['max']:.2f}",
                f"{stats['by_level'][level]['ssim']['mean']:.4f} ± {stats['by_level'][level]['ssim']['std']:.4f}",
                f"{stats['by_level'][level]['ssim']['min']:.4f}-{stats['by_level'][level]['ssim']['max']:.4f}",
                stats['by_level'][level]['psnr']['count']
            ])
        table_data.append(['TOTAL',
                           f"{stats['global']['psnr']['mean']:.2f} ± {stats['global']['psnr']['std']:.2f}",
                           f"{stats['global']['psnr']['min']:.2f}-{stats['global']['psnr']['max']:.2f}",
                           f"{stats['global']['ssim']['mean']:.4f} ± {stats['global']['ssim']['std']:.4f}",
                           f"{stats['global']['ssim']['min']:.4f}-{stats['global']['ssim']['max']:.4f}",
                           stats['global']['psnr']['count']
                           ])

        # log
        logger.info("\n" + tabulate(
            table_data,
            headers=[
                "Noise Level",
                "PSNR (dB)",
                "PSNR Range",
                "SSIM",
                "SSIM Range",
                "Samples"
            ],
            tablefmt="fancy_grid",
            floatfmt=("", ".2f", "", ".4f", "", "d")
        ))
        logger.info(f"Mean PSNR: {stats['global']['psnr']['mean']:.2f} dB")
        logger.info(f"Mean SSIM: {stats['global']['ssim']['mean']:.4f}")
        logger.info("=" * 60)

        return {
            'psnr': stats['global']['psnr']['mean'],
            'ssim': stats['global']['ssim']['mean']
        }

    def get_filename(self, idx):
        """Get the original file name (without noise level information)"""
        sample_idx = idx // len(self.NOISE_LEVELS)
        return self.sample_names[sample_idx]

    def __del__(self):
        """ensure hdf5 connection is closed"""
        if self._h5 is not None:
            self._h5.close()


@DeprecationWarning
@DATASET.register_module()
class MultiLevelRainHSIDatasetV0(Dataset):
    RAIN_TYPES = ['small', 'medium', 'heavy', 'storm']  # Four noise types are fixed

    def __init__(self,
                 data_root: str,
                 pipelines: Dict = None):
        """
        Args:
            data_root: The root directory of the dataset
        """
        self.root = data_root
        self.pipelines = Compose(pipelines) if pipelines else None

        # Check the directory structure
        self._validate_directory_structure()

        # Load all sample paths
        self.gt_paths = sorted(glob.glob(os.path.join(data_root, 'gt', '*.npy')))
        if not self.gt_paths:
            raise FileNotFoundError(f"No .npy files found in {os.path.join(data_root, 'gt')}")

        self.paths = self._build_paths_index()

    def _validate_directory_structure(self):
        """Verify that the directory structure is as expected"""
        required_dirs = ['gt'] + [os.path.join('rain', t) for t in self.RAIN_TYPES]
        for d in required_dirs:
            if not os.path.isdir(os.path.join(self.root, d)):
                raise RuntimeError(f"Missing required directory: {d}")

    def _build_paths_index(self) -> List[Dict[str, str]]:
        """Build a list of sample index dictionaries"""
        paths = []
        for gt_path in self.gt_paths:
            sample = {'gt_path': gt_path}

            # Obtain the corresponding noise file path
            filename, ext = os.path.basename(gt_path).split('.')
            for rain_type in self.RAIN_TYPES:
                rain_path = os.path.join(self.root, 'rain', rain_type, f'{filename}_{rain_type}.{ext}')
                if not os.path.exists(rain_path):
                    raise FileNotFoundError(f"Missing corresponding rain file: {rain_path}")
                sample['lq_path'] = rain_path
                paths.append(sample.copy())

        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.paths[idx]

        # 加载数据
        gt_path = sample['gt_path']
        lq_path = sample['lq_path']
        gt = np.load(gt_path)
        lq = np.load(lq_path)

        results = {
            'sample': lq,
            'target': gt,
            'index': idx
        }

        return self.pipelines(results) if self.pipelines else results

    def get_rain_types(self) -> List[str]:
        """获取支持的噪声类型列表"""
        return self.RAIN_TYPES.copy()
