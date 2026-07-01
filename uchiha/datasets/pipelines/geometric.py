import math
import numbers
from typing import Optional, Union, List, Tuple

import cv2
import numpy as np


def img_normalize(img, mean, std, to_rgb=True, scope='spatial', mode='standard'):
    """Normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.
        scope (str): Normalized scope.

    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    return img_normalize_(img, mean, std, to_rgb, scope, mode)


def img_normalize_(img, mean, std, to_rgb=True, scope='spatial', mode='standard'):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.
        scope (str): Normalized scope.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    channel = img.shape[-1]
    mean = np.float64(mean.reshape(1, -1)) if mean else None
    stdinv = 1 / np.float64(std.reshape(1, -1)) if std else None

    if to_rgb and channel == 3:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace

    if scope == 'spatial':
        if not mean:
            mean = np.float64(np.mean(img, axis=(0, 1)).reshape(1, -1))
        if not std:
            std = np.float64(np.std(img, axis=(0, 1)).reshape(1, -1))
            mask = std != 0
            stdinv = np.zeros_like(std)
            stdinv[mask] = 1 / std[mask]
        _min = np.float64(np.min(img, axis=(0, 1)).reshape(1, -1))
        _max = np.float64(np.max(img, axis=(0, 1)).reshape(1, -1))
    else:
        if not mean:
            mean = np.float32(np.mean(img, axis=2))[:, :, None]
            mean = np.tile(mean, (1, 1, channel))
        if not std:
            std = np.float32(np.std(img, axis=2))[:, :, None]
            std = np.tile(std, (1, 1, channel))
            stdinv = 1 / std
        _min = np.tile(np.float32(np.min(img, axis=2))[:, :, None], (1, 1, channel))
        _max = np.tile(np.float32(np.max(img, axis=2))[:, :, None], (1, 1, channel))

    if mode == 'standard':
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
    elif mode == 'minmax':
        cv2.subtract(img, _min, img)  # inplace
        cv2.multiply(img, 1 / (_max - _min), img)  # inplace
    else:
        raise NotImplementedError(f"normalize mode:{mode} is not supported yet")

    return img


def impad(img: np.ndarray,
          *,
          shape: Optional[Tuple[int, int]] = None,
          padding: Union[int, tuple, None] = None,
          pad_val: Union[float, List] = 0,
          padding_mode: str = 'constant') -> np.ndarray:
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on top/bottom and left/right
            respectively. If a tuple of length 4 is provided this is the
            padding for the top, bottom, left and right borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """
    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (height // 2, math.ceil(height / 2), width // 2, math.ceil(width / 2))

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }

    # check dim <= 512
    if img.shape[-1] > 512:
        img_patch1 = cv2.copyMakeBorder(img[:, :, :512], padding[0], padding[1], padding[2], padding[3],
                                        border_type[padding_mode],
                                        value=pad_val)
        img_patch2 = cv2.copyMakeBorder(img[:, :, 512:], padding[0], padding[1], padding[2], padding[3],
                                        border_type[padding_mode],
                                        value=pad_val)
        img = np.concatenate((img_patch1,img_patch2), axis=-1)
    else:

        img = cv2.copyMakeBorder(
            img,
            padding[0],
            padding[1],
            padding[2],
            padding[3],
            border_type[padding_mode],
            value=pad_val)

    return img


def impad_to_multiple(img: np.ndarray,
                      divisor: int,
                      pad_val: Union[float, List] = 0,
                      padding_mode: str = 'constant') -> np.ndarray:
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (Number | Sequence[Number]): Same as :func:`impad`.
        padding_mode: refer to impad

    Returns:
        ndarray: The padded image.
    """
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, shape=(pad_h, pad_w), pad_val=pad_val, padding_mode=padding_mode)
