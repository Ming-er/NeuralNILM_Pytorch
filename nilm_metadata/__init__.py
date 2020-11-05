from .object_concatenation import get_appliance_types, recursively_update_dict
from .convert_yaml_to_hdf5 import convert_yaml_to_hdf5, save_yaml_to_datastore
from nilm_metadata.version import version as __version__

import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'central_metadata', path)


