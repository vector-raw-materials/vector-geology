import omfvista
from dotenv import dotenv_values


def load_omf(env_name='PATH_TO_STONEPARK'):
    config = dotenv_values()
    path = config.get(env_name)
    omf = omfvista.load_project(path)
    return omf



def extend_box(extent, percentage=5):
    # Unpack the extent
    xmin, xmax, ymin, ymax, zmin, zmax = extent

    # Calculate the increments for each dimension
    x_increment = (xmax - xmin) * (percentage / 100.0) / 2
    y_increment = (ymax - ymin) * (percentage / 100.0) / 2
    z_increment = (zmax - zmin) * (percentage / 100.0) / 2

    # Extend the extents
    new_xmin = xmin - x_increment
    new_xmax = xmax + x_increment
    new_ymin = ymin - y_increment
    new_ymax = ymax + y_increment
    new_zmin = zmin - z_increment
    new_zmax = zmax + z_increment

    return [new_xmin, new_xmax, new_ymin, new_ymax, new_zmin, new_zmax]
