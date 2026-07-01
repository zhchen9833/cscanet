import torch


def validate(epoch, dataloader, model, writer, metric, device):
    """ validation during training

    Args:
        epoch (int): the training epoch when the validation frequency is reached
        dataloader (torch.utils.data.Dataloader): validation set's dataloader
        model (torch.nn.Module): model built from configuration file
        writer (SummaryWriter): tensorboard-based loggers
        metric (str): model's performance metrics for specific tasks
        device (torch.device): device to run the model

    Returns:
        results (dict): Dictionary, the key is the name corresponding to cls or ele,
        and the value is the result of the model output.

    """
    dataset = dataloader.dataset
    targets = []
    preds = []
    indexes = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            # data
            sample = data['sample'].to(device, non_blocking=True)
            # forward
            pred = model(sample)

            targets.append(data['target'].numpy())
            indexes.append(data['index'].numpy())
            preds.append(pred.cpu().numpy())

    # evaluate
    results = dataset.evaluate(preds, targets, metric, indexes)

    # log
    if isinstance(results, (int, float, complex)):
        writer.add_scalar(f'{metric}', results, epoch)
    elif isinstance(results, dict):
        for ele in results:
            writer.add_scalar(ele, results[ele], epoch)
    elif isinstance(results, (list, tuple, str, bytes)):
        pass
    else:
        raise NotImplementedError(f'results type:{type(results)} not supported yet !')

    return results
