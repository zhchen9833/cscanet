from ..builder import PIPELINES
from .geometric import impad, impad_to_multiple


@PIPELINES.register_module()
class Pad:
    """ Pad the image.

    There are two padding modes:
    (1) pad to a fixed size
    (2) pad to the minimum size that is divisible by some number.
    there are four ways to handle edges:
    (1) constant: pad with a constant -> the given pad_val
    (2) edge: pad with the last value at the edge of the image.
    (3) reflect: pads with reflection of image without repeating the last value on the edge.
        For example: padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
        will result in [3, 2, 1, 2, 3, 4, 3, 2].
    (4) symmetric: pads with reflection of image repeating the last value on the edge.
        For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
        will result in [2, 1, 1, 2, 3, 4, 4, 3]
    Added key is 'pad_cfg'.

    Args:
        size (tuple, optional): Fixed padding size -> mode(1).
        size_divisor (int, optional): The divisor of padded size -> mode(2).
        pad_val (dict, optional): A dict for padding value, Default: `0`.
        mode (str): the way to handle edges
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 mode='constant'):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.mode = mode

        assert size is not None or size_divisor is not None, \
            'only one of size and size_divisor should be valid'

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val
        if 'sample' in results:
            key = 'sample'
        else:
            key = None

        results['pad_cfg'] = dict(ori_shape=results[key].shape, pad_shape=None)

        if self.size is not None:
            padded_img = impad(
                results[key], shape=self.size, pad_val=pad_val, padding_mode=self.mode)
        elif self.size_divisor is not None:
            padded_img = impad_to_multiple(
                results[key], self.size_divisor, pad_val=pad_val, padding_mode=self.mode)
        else:
            padded_img = None
        results[key] = padded_img
        results['pad_cfg']['pad_shape'] = padded_img.shape

    def __call__(self, results):
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            results (dict): Updated result dict,'pad_cfg' key is added into result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str
