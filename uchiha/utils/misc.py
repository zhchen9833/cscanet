import os.path
from typing import Union


def strings_to_list(x: Union[str, dict]) -> Union[list, dict]:
    """ string(s) --> list

    Args:
        x (str | dict): A single string or a dict containing strings

    Returns:
        list: Converted list(value of dict) of containing strings
    """
    if isinstance(x, dict):
        for key, value in x.items():
            if isinstance(value, dict):
                strings_to_list(value)
                continue
            elif isinstance(value, str):
                splits = value.split(',')
            else:
                continue
            if len(splits) > 1:
                new_value = [int(num) if num.strip().isdigit() else num.strip() for num in splits]
                x[key] = new_value
    else:
        splits = x.split(',')
        return [int(num) if num.strip().isdigit() else num.strip() for num in splits]
    return x


def get_extension(path):
    """ Get the extension of a file

    Args:
        path (str): File path

    Returns:
        str: The extension of the file
    """
    basename = os.path.basename(path)
    return basename.split('.')[-1]
