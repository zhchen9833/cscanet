from tabulate import tabulate
from torch import nn


def count_parameters(module):
    """ Calculate the parameters of the model.

    Args:
        module (nn.Module): The model that require counting parameter quantities

    Returns:
        int: Parameter quantities of the model
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def unwrap_model(model):
    """unwrap the model of the ddp dataparallel"""
    if hasattr(model, 'module'):
        return model.module
    return model


def format_param_count(n):
    """format parameter quantity in k m units。"""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    else:
        return str(n)


def log_model_parameters(model: nn.Module, logger, max_depth: int = 1):
    """
    Record the parameter structure of the model, display the total number and proportion of parameters in table form, and sort by parameter amount.

    Args:
        model (nn.Module): the model object
        logger: the logger
        max_depth (int): the maximum level of presentation
    """

    def collect_named_param_info(_model, _max_depth):
        _rows = []
        total_params = count_parameters(_model)
        stack = [("", _model, 0)]

        while stack:
            prefix, module, depth = stack.pop(0)
            if depth > _max_depth:
                continue

            name = prefix if prefix != "" else _model.__class__.__name__
            n_params = count_parameters(module)
            percent = f"{(n_params / total_params) * 100:.2f}%" if total_params > 0 else "0.00%"

            _rows.append({
                "depth": depth,
                "name": name,
                "type": module.__class__.__name__,
                "params": n_params,
                "percent": percent
            })

            if depth < _max_depth:
                for child_name, child in module.named_children():
                    full_name = f"{prefix}.{child_name}" if prefix else child_name
                    stack.append((full_name, child, depth + 1))

        return _rows, total_params

    logger.info("== Start analyzing model parameters ==")
    rows, total = collect_named_param_info(model, max_depth)

    rows.sort(key=lambda x: x["params"], reverse=True)

    table_data = [
        [r["depth"], r["name"], r["type"], format_param_count(r["params"]), r["percent"]]
        for r in rows
    ]

    table = tabulate(
        table_data,
        headers=["Depth", "Module Name", "Type", "Params", "Percent"],
        tablefmt="fancy_grid"
    )

    logger.info("\n" + table)
    logger.info(f"== Total parameters: {format_param_count(total)} ==")
