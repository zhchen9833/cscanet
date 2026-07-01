import random

import cv2

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomRotation:
    def __init__(self, angles=None, prob=0.5, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101):
        """
        Args:
            angles (tuple): rotation angle range for example 30 30
            prob (float): probability of triggering enhancement
            interpolation (int): interpolation method（cv2.INTER_NEAREST / INTER_LINEAR / INTER_CUBIC）
            border_mode (int): boundary fill method（cv2.BORDER_CONSTANT / BORDER_REPLICATE / BORDER_REFLECT_101）
        """
        if angles is None:
            angles = [-90, 90]
        self.angles = angles
        self.p = prob
        self.interpolation = interpolation
        self.border_mode = border_mode

    def __call__(self, data):
        if random.random() < self.p:
            sample = data['sample']
            target = data['target']

            h, w = sample.shape[:2]
            center = (w / 2, h / 2)

            # randomly generate rotation angle
            angle = random.choice(self.angles)
            if isinstance(angle, int):
                angle = float(angle)

            # compute the rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

            # applying affine transformation
            if len(sample.shape) == 3:  # HWC
                sample_rotated = cv2.warpAffine(sample, M, (w, h), flags=self.interpolation,
                                                borderMode=self.border_mode)
            else:  # HW
                sample_rotated = cv2.warpAffine(sample, M, (w, h), flags=self.interpolation,
                                                borderMode=self.border_mode)

            if len(target.shape) == 3:  # HWC
                target_rotated = cv2.warpAffine(target, M, (w, h), flags=self.interpolation,
                                                borderMode=self.border_mode)
            else:  # HW
                target_rotated = cv2.warpAffine(target, M, (w, h), flags=self.interpolation,
                                                borderMode=self.border_mode)

            data['sample'] = sample_rotated
            data['target'] = target_rotated

        return data
