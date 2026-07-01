import os.path
import shutil

import yaml

from uchiha.utils.misc import get_extension


class Config:
    # TODO 配置继承
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = Config(**value)
            setattr(self, key, value)

    def __repr__(self):
        return self._repr_recursive(self, 0)

    def _repr_recursive(self, d, indent_level):
        """print"""
        indent = ' ' * (indent_level * 2)
        repr_str = ''
        for key, value in d.__dict__.items():
            if isinstance(value, Config):
                repr_str += f"{indent}{key}:\n"
                repr_str += self._repr_recursive(value, indent_level + 1)
            else:
                repr_str += f"{indent}{key}: {value}\n"
        return repr_str

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value = value.to_dict()
            result[key] = value
        return result


def load_config(file_path):
    ext = get_extension(file_path)

    if ext == 'yaml' or ext == 'yml':
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
    else:
        raise NotImplementedError('not supported yet')

    work_dir = config_dict['work_dir']
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)
    shutil.copy(file_path, work_dir)

    return Config(**config_dict)


if __name__ == '__main__':
    yaml_path = 'configs/s_e512_wo-w.yaml'
    config = load_config(yaml_path)
    print(config)
    print(config.data.dataloader.to_dict())
