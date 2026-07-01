import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset
from terminaltables import AsciiTable

from .builder import DATASET
from .pipelines import Compose
from .load import read_pts, read_txt, read_npy
from ..utils import print_log, get_root_logger, strings_to_list


def tensor2np(x):
    """ tensor --> ndarray

    Args:
        x (Tensor): Input tensor data

    Returns:
        ndarray: converted ndarray data
    """
    if isinstance(x, list):
        for idx, data in enumerate(x):
            x[idx] = tensor2np(data)
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return x


def regression_eval(preds, targets, elements, metric='MAE'):
    """ evaluation of regression tasks

    Args:
        preds (Tensor): Prediction of the model
        targets (Tensor): GT data
        elements (List[str]): Elements that need to be predicted
        metric (str): Evaluation metric. Default: MAE.

    Returns:
        dict: evaluation results for each element
    """

    results = {element: 0.0 for element in elements}

    preds = np.vstack(tensor2np(preds))
    targets = np.vstack(tensor2np(targets))
    metrics = compute_metric(preds, targets, metric)
    if len(results) == 1:
        results[elements[0]] = metrics[0]
    else:
        for idx, element in enumerate(elements):
            results[element] = metrics[idx]

    print_metrics(results, metric)

    return results


def compute_metric(pred, target, metric='MAE'):
    """ given metric, calculation result

    Args:
        pred (ndarray): Prediction
        target (ndarray): GT
        metric (str): Evaluation metric. Default: MAE.

    Returns:
        ndarray: Calculation results
    """
    if metric == 'MAE':
        return mean_absolute_error(pred, target)
    elif metric == 'R2':
        return 1 - (np.sum((target - pred) ** 2, axis=0) / np.sum((target - np.mean(target)) ** 2, axis=0))
        # return (np.sum((pred - np.mean(target)) ** 2) / np.sum((target - np.mean(target)) ** 2))
    else:
        raise NotImplementedError(f'metric:{metric} is not supported yet')


def print_metrics(result, metric):
    """ print metrics in the console

    Args:
        result (dict): Evaluation results for each element .
        metric (str): Evaluation metric.

    """
    header = ['Element', f'{metric}']
    table_data = [header]
    for key, value in result.items():
        row_data = [key, f'{value:.6f}']
        table_data.append(row_data)

    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    table.justify_columns[1] = 'center'

    logger = get_root_logger()
    print_log('\n' + table.table, logger=logger)


@DATASET.register_module()
class SoilDataset2d(Dataset):
    """ the dataset for hyperspectral images

    Args:
        data_root (str): the root directory where the dataset is stored
        gt_path (str): path of GT data
        elements (str): Elements to be predicted, a comma-separated string is parsed into a list
        pipelines (Sequence[dict]): Data processing flow: a sequence
            each element is different data processing configuration information
    """

    ELEMENTS = ['Zn', 'Others']

    def __init__(self, data_root, gt_path, elements=None, pipelines=None):

        self.data_root = data_root
        self.spectral_data = read_npy(data_root)
        self.gt = read_txt(gt_path)
        self.ELEMENTS = self.get_elements(elements)

        # postprocess
        self.pipelines = Compose(pipelines)

    def __getitem__(self, index):
        results = dict(
            sample=self.spectral_data[index],
            target=self.gt[index],
            index=index
        )
        return self.pipelines(results)

    def __len__(self):
        return len(self.spectral_data)

    def get_elements(self, elements):
        """ get the elements to be predicted

        Gets a list of containing elements based on the provided string separated by commas

        Args:
            elements (str): Elements to be predicted

        Returns:
            element_names (list):  the list containing elements that need to be predicted

        """
        if elements is None:
            return self.ELEMENTS
        if isinstance(elements, str):

            element_names = strings_to_list(elements)

        elif isinstance(elements, (tuple, list)):
            element_names = elements
        else:
            raise ValueError(f'Unsupported type {type(elements)} of elements.')

        return element_names

    def evaluate(self, preds, targets=None, metric='MAE', indexes=None):
        """ evaluation of predicted values

        Args:
            preds (Tensor): output prediction of the model
            targets (Tensor): the true value of the predicted element
            metric (str): evaluation metric. Default: 'MAE'
            indexes (int): index of original data
        Returns:
            results (dict): evaluation results for each element

        """
        elements = self.ELEMENTS
        if targets is None:
            targets = self.gt

        results = regression_eval(preds, targets, elements, metric)

        return results


@DATASET.register_module()
class SoilDataset1d(Dataset):
    """ the dataset for hyperspectral sequence-data

    Args:
        data_root (str): the root directory where the dataset is stored
        gt_path (str): path of GT data
        elements (str): Elements to be predicted, a comma-separated string is parsed into a list
        pipelines (Sequence[dict]): Data processing flow: a sequence
            each element is different data processing configuration information
    """

    ELEMENTS = ['Zn', 'Subs']

    def __init__(self, data_root, gt_path, elements=None, pipelines=None):
        self.data_root = data_root
        self.spectral_data = read_pts(data_root, is1d=True)
        self.gt = read_txt(gt_path)
        self.ELEMENTS = self.get_elements(elements)

        self.pipelines = Compose(pipelines) if pipelines else None

    def __getitem__(self, index):
        results = dict(
            sample=self.spectral_data[index],
            target=self.gt[index],
            index=index
        )
        return self.pipelines(results) if self.pipelines else results

    def __len__(self):
        return len(self.spectral_data)

    def get_elements(self, elements):
        """ get the elements to be predicted

        Gets a list of containing elements based on the provided string separated by commas

        Args:
            elements (str): Elements to be predicted

        Returns:
            element_names (list):  the list containing elements that need to be predicted

        """
        if elements is None:
            return self.ELEMENTS
        if isinstance(elements, str):

            element_names = strings_to_list(elements)

        elif isinstance(elements, (tuple, list)):
            element_names = elements
        else:
            raise ValueError(f'Unsupported type {type(elements)} of elements.')

        return element_names

    def evaluate(self, preds, targets=None, metric='MAE'):
        """ evaluation of predicted values

        Args:
            preds (Tensor): output prediction of the model
            targets (Tensor): the true value of the predicted element
            metric (str): evaluation metric. Default: 'MAE'

        Returns:
            results (dict): evaluation results for each element

        """
        elements = self.ELEMENTS
        if targets is None:
            targets = self.gt

        results = regression_eval(preds, targets, elements, metric)

        return results
