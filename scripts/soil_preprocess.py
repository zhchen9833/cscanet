import os
from os.path import join

import numpy as np
import torch


def txt2list(sample):
    txts = os.listdir(sample)
    txts = sorted(txts, key=lambda x: int(x.split('.')[0]))
    data = []
    for txt in txts:
        tmp = []
        txt = join(sample, txt)
        with open(txt, 'r', encoding="GBK") as f:
            lines = f.readlines()
            lines = lines[3:]
            for line in lines:
                line = line.strip()
                reflectivity = float(line.split(' ')[-1]) / 10000
                tmp.append(reflectivity)

        data.append(tmp)

    return data


def read(sample_dir, save_dir, to_tensor=True):
    samples = os.listdir(sample_dir)
    for sample_idx in samples:
        sample = join(sample_dir, sample_idx)
        data = txt2list(sample)
        if to_tensor:
            data = torch.tensor(data)
            length, channel = data.shape
            data = data.view(int(length ** 0.5), int(length ** 0.5), channel)
            torch.save(data, join(save_dir, f'{sample_idx}.pt'))
        else:
            data = np.array(data)
            # data = data.transpose(0, 1).reshape(10, 33, -1)
            length, channel = data.shape
            data = data.reshape(int(length ** 0.5), int(length ** 0.5), channel)
            np.save(join(save_dir, f'{sample_idx}.npy'), data)


def pt2npy(sample_dir):
    samples = os.listdir(sample_dir)
    for sample_idx in samples:
        sample = join(sample_dir, sample_idx)
        data = torch.load(sample).unsqueeze(0)
        data = data.numpy()
        np.save(os.path.join(sample_dir,sample_idx.replace('pt', 'npy')), data)


if __name__ == '__main__':
    # sample_dir = '/home/disk1/ZR/datasets/soil_pred'
    # save_dir = '/data/spectral_01/train/reflectivity'
    # read(sample_dir, save_dir, to_tensor=False)

    src = 'data/dataset'
    des = 'data/npy/train/reflectivity'
    read(src,des, to_tensor=False)
