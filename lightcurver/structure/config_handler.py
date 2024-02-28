import yaml


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def dump_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
