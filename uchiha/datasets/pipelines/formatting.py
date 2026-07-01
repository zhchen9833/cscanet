import torch

from uchiha.datasets.builder import PIPELINES


@PIPELINES.register_module()
class EasyToTensor:
    """ convert data to tensor

    The default input data is in `B H W C` form, converting it to `B C H W` form

    Args:
        mode (str): data order form
    """
    def __init__(self, mode='CHW'):

        self.mode = mode

    def __call__(self, results):
        """Call function to convert the type of the data.

        Args:
            results (dict): Result dict from pipeline.

        Returns:
            results (dict): converted results
        """
        assert 'sample' in results and 'target' in results, 'sample and target should be loaded to results'

        if self.mode == 'CHW':
            results['sample'] = torch.Tensor(results['sample']).permute(2, 0, 1)
            results['target'] = torch.Tensor(results['target']).permute(2, 0, 1)
        else:
            results['sample'] = torch.Tensor(results['sample'])
            results['target'] = torch.Tensor(results['target'])

        return results
