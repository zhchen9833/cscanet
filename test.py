import argparse
from datetime import datetime
from os.path import join

from thop import profile, clever_format
import torch
from torch import nn

from uchiha.apis import set_random_seed, complex_test
from uchiha.apis.inference_dehaze import remove_module_prefix
from uchiha.datasets.builder import build_dataset, build_dataloader
from uchiha.models.builder import build_model
from uchiha.utils import load_config, get_root_logger


def get_info(model, inp):
    model.eval()

    macs, params = profile(model, inputs=(inp,))
    macs, params = clever_format([macs, params], "%.3f")

    return macs, params


def get_checkpoint_path(cfg):
    return getattr(cfg, 'checkpoint', None)


def get_test_dataset_name(cfg):
    dataset_cfg = cfg.data.test.dataset
    return getattr(dataset_cfg, 'dataset_name', 'HD')


def get_test_metric_options(cfg):
    explicit_no_reference = getattr(cfg, 'no_reference', None)
    if explicit_no_reference is None:
        dataset_name = get_test_dataset_name(cfg)
        no_reference = dataset_name == 'HDD'
    else:
        no_reference = explicit_no_reference

    return (
        no_reference,
        getattr(cfg, 'include_no_reference', False),
    )


def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--seed', type=int, default=49)
    args_parser.add_argument('--config', type=str, default='configs/hsi_dehaze/HD/test/D3.yaml')
    args_parser.add_argument('--gpu_ids', nargs='+', default=['0'])
    args_parser.add_argument('--info', type=int, default=1)

    return args_parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # log: tensorboard & logger
    work_dir = cfg.work_dir
    log_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

    logger = get_root_logger(log_file=join(f'{work_dir}/logs', f'{log_time}.log'))
    logger.info(f'Config:\n{cfg}')

    # random seed
    set_random_seed(args.seed)
    logger.info(f'set random seed= {args.seed}')

    # dataset & dataloader
    testset = build_dataset(cfg.data.test.dataset.to_dict(), phase='test')
    testloader = build_dataloader(testset, cfg.data.test.dataloader.to_dict(), phase='test')

    # model
    model = build_model(cfg.model.to_dict())
    checkpoint_path = get_checkpoint_path(cfg)
    checkpoint = torch.load(checkpoint_path) if checkpoint_path else None
    gpu_ids = [int(i) for i in args.gpu_ids]
    if len(gpu_ids) > 1:
        torch.cuda.set_device(gpu_ids[0])  # 当前上下文绑定主卡
        model = nn.DataParallel(model, device_ids=gpu_ids).cuda(gpu_ids[0])
        device = torch.device(f'cuda:{gpu_ids[0]}')
        logger.info(f'Using GPUs: {gpu_ids}')
        if checkpoint is not None:
            model.load_state_dict(checkpoint['state_dict'])
    else:
        device_id = gpu_ids[0]
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)  # 当前上下文绑定该卡
        model = model.to(device)
        logger.info(f'Using single GPU: {device}')
        if checkpoint is not None:
            model.load_state_dict(remove_module_prefix(checkpoint['state_dict']))
    if checkpoint is not None:
        logger.info(f'checkpoint:{checkpoint_path} was loaded successfully!')
    else:
        logger.info('no checkpoint was configured; running the model directly.')

    # model info
    if args.info > 0 and getattr(cfg, 'profile_model', True):
        num_bands = getattr(cfg, 'num_bands', 305)
        info_size = getattr(cfg, 'info_size', 512)
        macs, params = get_info(model, inp=torch.randn(1, num_bands, info_size, info_size).cuda())

        logger.info(f"FLOPs: {macs}")
        logger.info(f"Parameters: {params}")
    elif args.info > 0:
        logger.info('model profiling was skipped by config.')

    # train & val
    logger.info('start testing...')

    # evaluate
    no_reference, include_no_reference = get_test_metric_options(cfg)
    complex_test(dataloader=testloader,
                 model=model,
                 device=device,
                 no_reference=no_reference,
                 include_no_reference=include_no_reference)


if __name__ == '__main__':
    main()
