import omfvista
from dotenv import dotenv_values


def load_omf(env_name='PATH_TO_STONEPARK'):
    config = dotenv_values()
    path = config.get(env_name)
    omf = omfvista.load_project(path)
    return omf

