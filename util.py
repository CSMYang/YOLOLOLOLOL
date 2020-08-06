"""
The helper functions for our YOLO model.
"""


def get_model_from_config(name):
    """
    :param name: The file path for the configuation file.
    Get the parameters of each layer of the neural network from the config file based on the given path name.
    """
    module_params = []
    module = open(name, 'r')
    lines = module.read().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        elif line[0] == '[':
            layer = dict()
            layer['type'] = line[1:-1]
            module_params.append(layer)
        else:
            key, value = line.split("=")
            module_params[-1][key.strip()] = value.strip()
    return module_params
