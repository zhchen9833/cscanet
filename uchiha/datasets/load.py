import re
from os import listdir
from os.path import join, isdir

import numpy as np
import torch


def read_pts(data_root, is1d=False):
    name_list = listdir(data_root)
    name_list = sorted(name_list, key=lambda x: int(x.split('.')[0]))
    path_list = [join(data_root, name) for name in name_list]
    if is1d:
        data = [torch.load(path) for path in path_list]
    else:
        data = [torch.load(path).permute(2, 0, 1) for path in path_list]
    return data


def read_txt(path):
    def extract_numbers(input_string):
        # remove extra tabs and spaces
        cleaned_string = re.sub(r'[\t\s]+', ' ', input_string.strip())
        # Extract all numbers (including integers and floating point numbers)
        numbers = re.findall(r'-?\d+(?:\.\d+)?', cleaned_string)
        # convert to a list of floating point numbers
        number_list = [float(num) for num in numbers]
        return number_list
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            # data.append(np.array([float(num) for num in line.strip().split(' ')]))
            data.append(np.array(extract_numbers(line)))

    return data


def read_npy(path):
    if isdir(path):
        name_list = listdir(path)
        name_list = sorted(name_list, key=lambda x: int(x.split('.')[0]))
        path_list = [join(path, name) for name in name_list]

        data = [np.load(path) for path in path_list]
    else:
        data = np.load(path)

    return data
