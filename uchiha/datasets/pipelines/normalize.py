import numpy as np

from uchiha.datasets.builder import PIPELINES
from uchiha.datasets.pipelines.geometric import img_normalize


@PIPELINES.register_module()
class Normalize:
    """Normalize the image.

    When the mean and standard deviation are given,the data will be normalized according to the given data,
    otherwise the mean and standard deviation of the input data will be calculated for normalization.
    There are two normalization modes:
    (1) standardize: (x-mean)/std
    (2) normalize(minmax): (x-min)/(max-min)
    There are two normalization scopes:
    (1) spatial: normalize the spatial data for each channel
    (2) channel: normalize the channel data for each pixel in space
    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels. Default: None.
        std (sequence): Std values of 3 channels. Default: None.
        to_rgb (bool): Whether to convert the image from BGR to RGB. Default: true.
        mode (str): normalization mode.
        scope (str): normalization scope
    """

    def __init__(self, mean=None, std=None, to_rgb=True, scope='spatial', mode='standard'):

        self.mean = np.array(mean, dtype=np.float32) if mean else None
        self.std = np.array(std, dtype=np.float32) if std else None
        self.to_rgb = to_rgb
        self.scope = scope
        self.mode = mode

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from pipeline.

        Returns:
            results (dict): Normalized results, 'img_norm_cfg' key is added into result dict.
        """
        results['sample'] = img_normalize(results['sample'], self.mean, self.std,
                                          self.to_rgb, self.scope, self.mode)
        if self.mean and self.std:
            results['norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_rgb=self.to_rgb, scope=self.scope)
        else:
            results['norm_cfg'] = dict(
                mean='auto', std='auto', to_rgb=self.to_rgb, scope=self.scope)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str
