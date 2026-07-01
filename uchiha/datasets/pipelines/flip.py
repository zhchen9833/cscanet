import random

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomFlip:
    def __init__(self, probs=None, directions=None):
        """
        Args:
            probs (list): The probability weights of the four flips, in the order:
[Original probability, horizontal flip probability, vertical flip probability, diagonal flip probability]
        """
        if directions is None:
            directions = ['original', 'horizontal', 'vertical', 'diagonal']
        if probs is None:
            probs = [0.5, 0.25, 0.25, 0.0]

        self.p = probs

        # normalized probability
        total = sum(probs)
        self.norm_p = [x / total for x in probs]

        # building a choice space
        self.directions = directions

        # Cumulative probability is used for random selection
        cum = 0.0
        self.cum_probs = []
        for probs in self.norm_p:
            cum += probs
            self.cum_probs.append(cum)

    def _apply_flip(self, img, flip_type):
        """perform specific flip operations"""
        if flip_type == 'original':
            return img
        elif flip_type == 'horizontal':
            return img[:, ::-1, ...].copy()  # HWC / HW
        elif flip_type == 'vertical':
            return img[::-1, :, ...].copy()
        elif flip_type == 'diagonal':
            return img[:, ::-1, ...][::-1, :, ...].copy()
        else:
            raise ValueError(f"Unknown flip type: {flip_type}")

    def __call__(self, data):
        sample = data['sample']
        target = data['target']

        # use cumulative probability to select flip type
        rand_val = random.random()
        selected = self.directions[-1]
        for i, cp in enumerate(self.cum_probs):
            if rand_val < cp:
                selected = self.directions[i]
                break

        # flip
        sample = self._apply_flip(sample, selected)
        target = self._apply_flip(target, selected)

        data['sample'] = sample
        data['target'] = target

        return data
