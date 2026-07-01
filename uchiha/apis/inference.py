import h5py
import torch
from tqdm import tqdm


def simple_inference_with_h5(dataloader, model, h5_file):
    """ inference

    Args:
        dataloader (torch.utils.data.Dataloader): validation set's dataloader
        model (torch.nn.Module): model built from configuration file
        h5_file (str): The file name of the saved h5 file

    """
    dataset = dataloader.dataset
    model.eval()

    pbar = tqdm(dataloader, desc="inference...", total=len(dataloader))

    with h5py.File(h5_file, 'w') as hf:
        lq_group = hf.create_group('lq')
        restored_group = hf.create_group('restored')
        gt_group = hf.create_group('gt')
        with torch.no_grad():
            for idx, data in enumerate(pbar):
                # data
                sample = data['sample'].cuda()
                # forward
                pred = model(sample)

                name = dataset.get_filename(data['index'].item())
                level = dataset.get_noise_level(data['index'].item())
                gt_group.create_dataset(f'{name}_{level}', data=data['target'].squeeze().numpy(), compression='lzf')
                restored_group.create_dataset(f'{name}_{level}', data=pred.squeeze().cpu().numpy(), compression='lzf')
                lq_group.create_dataset(f'{name}_{level}', data=sample.squeeze().cpu().numpy(), compression='lzf')

                pbar.set_postfix(iter=f"{idx + 1}/{len(dataloader)}")

    pbar.close()


def simple_inference(dataloader, model):
    """ inference

    Args:
        dataloader (torch.utils.data.Dataloader): validation set's dataloader
        model (torch.nn.Module): model built from configuration file

    """
    dataset = dataloader.dataset
    model.eval()

    pbar = tqdm(dataloader, desc="inference...", total=len(dataloader))

    samples = []
    preds = []
    targets = []
    with torch.no_grad():
        for idx, data in enumerate(pbar):
            # data
            sample = data['sample'].cuda()
            # forward
            pred = model(sample)

            sample = sample.cpu().numpy()
            pred = pred.cpu().numpy()
            target = data['target'].cpu().numpy()
            B, C, H, W = sample.shape
            for i in range(B):
                samples.append(sample[i, :, :, :])
                targets.append(target[i, :, :, :])
                preds.append(pred[i, :, :, :])

            pbar.set_postfix(iter=f"{idx + 1}/{len(dataloader)}")

    pbar.close()
