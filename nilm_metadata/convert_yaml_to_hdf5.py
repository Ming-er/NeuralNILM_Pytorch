from __future__ import print_function, division
import yaml
import pandas as pd
from os.path import isdir, isfile, join, splitext
from os import listdir
from sys import stderr
from copy import deepcopy
from six import iteritems
from .object_concatenation import get_appliance_types


class NilmMetadataError(Exception):
    pass


def convert_yaml_to_hdf5(yaml_dir, hdf_filename):
    """Converts a NILM Metadata YAML instance to HDF5.

    Also does a set of sanity checks on the metadata.

    Parameters
    ----------
    yaml_dir : str
        Directory path of all *.YAML files describing this dataset.
    hdf_filename : str
        Filename and path of output HDF5 file.  If file exists then will
        attempt to append metadata to file.  If file does not exist then
        will create it.
    """

    assert isdir(yaml_dir)
    store = pd.HDFStore(hdf_filename, 'a')

    # Load Dataset and MeterDevice metadata
    metadata = _load_file(yaml_dir, 'dataset.yaml')
    meter_devices = _load_file(yaml_dir, 'meter_devices.yaml')
    metadata['meter_devices'] = meter_devices
    store.root._v_attrs.metadata = metadata

    # Load buildings
    building_filenames = [fname for fname in listdir(yaml_dir)
                          if fname.startswith('building')
                          and fname.endswith('.yaml')]

    for fname in building_filenames:
        building = splitext(fname)[0]  # e.g. 'building1'
        try:
            group = store._handle.create_group('/', building)
        except:
            group = store._handle.get_node('/' + building)
        building_metadata = _load_file(yaml_dir, fname)
        elec_meters = building_metadata['elec_meters']
        _deep_copy_meters(elec_meters)
        _set_data_location(elec_meters, building)
        _sanity_check_meters(elec_meters, meter_devices)
        _sanity_check_appliances(building_metadata)
        group._f_setattr('metadata', building_metadata)

    store.close()
    print("Done converting YAML metadata to HDF5!")


def save_yaml_to_datastore(yaml_dir, store):
    """Saves a NILM Metadata YAML instance to a NILMTK datastore.

    Parameters
    ----------
    yaml_dir : str
        Directory path of all *.YAML files describing this dataset.
    store : DataStore
        DataStore object
    """

    assert isdir(yaml_dir)

    # Load Dataset and MeterDevice metadata
    metadata = _load_file(yaml_dir, 'dataset.yaml')
    print("Loaded metadata")
    meter_devices = _load_file(yaml_dir, 'meter_devices.yaml')
    metadata['meter_devices'] = meter_devices
    store.save_metadata('/', metadata)

    # Load buildings
    building_filenames = [fname for fname in listdir(yaml_dir)
                          if fname.startswith('building')
                          and fname.endswith('.yaml')]

    for fname in building_filenames:
        building = splitext(fname)[0]  # e.g. 'building1'
        building_metadata = _load_file(yaml_dir, fname)
        elec_meters = building_metadata['elec_meters']
        _deep_copy_meters(elec_meters)
        _set_data_location(elec_meters, building)
        _sanity_check_meters(elec_meters, meter_devices)
        _sanity_check_appliances(building_metadata)
        store.save_metadata('/'+building, building_metadata)

    store.close()
    print("Done converting YAML metadata to HDF5!")


def _load_file(yaml_dir, yaml_filename):
    yaml_full_filename = join(yaml_dir, yaml_filename)
    if isfile(yaml_full_filename):
        with open(yaml_full_filename, 'rb') as fh:
            return yaml.safe_load(fh)
    else:
        print(yaml_full_filename, "not found.", file=stderr)


def _deep_copy_meters(elec_meters):
    for meter_instance, meter in iteritems(elec_meters):
        elec_meters[meter_instance] = deepcopy(meter)


def _set_data_location(elec_meters, building):
    """Goes through each ElecMeter in elec_meters and sets `data_location`.
    Modifies `elec_meters` in place.

    Parameters
    ----------
    elec_meters : dict of dicts
    building : string e.g. 'building1'
    """
    for meter_instance in elec_meters:
        data_location = '/{:s}/elec/meter{:d}'.format(building, meter_instance)
        elec_meters[meter_instance]['data_location'] = data_location


def _sanity_check_meters(meters, meter_devices):
    """
    Checks:
    * Make sure all meter devices map to meter_device keys
    * Makes sure all IDs are unique
    """
    if len(meters) != len(set(meters)):
        raise NilmMetadataError("elec_meters not unique")

    for meter_instance, meter in iteritems(meters):
        assert meter['device_model'] in meter_devices


def _sanity_check_appliances(building_metadata):
    """
    Checks:
    * Make sure we use proper NILM Metadata names.
    * Make sure there aren't multiple appliance types with same instance
    """
    appliances = building_metadata['appliances']
    appliance_types = get_appliance_types()
    building_instance = building_metadata['instance']
    REQUIRED_KEYS = ['type', 'instance', 'meters']

    for appliance in appliances:
        if not isinstance(appliance, dict):
            raise NilmMetadataError(
                "Appliance '{}' is {} when it should be a dict."
                .format(appliance, type(appliance)))

        # Generate string for specifying which is the problematic
        # appliance for error messages:
        appl_string = ("ApplianceType '{}', instance '{}', in building {:d}"
                       .format(appliance.get('type'),
                               appliance.get('instance'),
                               building_instance))

        # Check required keys are all present
        for key in REQUIRED_KEYS:
            if key not in appliance:
                raise NilmMetadataError("key '{}' missing for {}"
                                        .format(key, appl_string))

        appl_type = appliance['type']

        # check all appliance names are valid
        if appl_type not in appliance_types:
            raise NilmMetadataError(
                appl_string + " not in appliance_types."
                "  In other words, '{}' is not a recognised appliance type."
                .format(appl_type))

        # Check appliance references valid meters
        meters = appliance['meters']
        if len(meters) != len(set(meters)):
            msg = "In {}, meters '{}' not unique.".format(appl_string, meters)
            raise NilmMetadataError(msg)

        for meter in meters:
            if meter != 0 and meter not in building_metadata['elec_meters']:
                msg = ("In ({}), meter '{:d}' is not in"
                       " this building's 'elec_meters'"
                       .format(appl_string, meter))
                raise NilmMetadataError(msg)

    # Check list of instances for each appliance is valid.
    appliance_instances = {}
    for appliance in appliances:
        appl_type = appliance['type']
        instances = appliance_instances.setdefault(appl_type, [])
        instances.append(appliance['instance'])

    for appliance_type, instances in iteritems(appliance_instances):
        instances.sort()
        correct_instances = list(range(1, len(instances)+1))
        if instances != correct_instances:
            msg = ("In building {:d}, appliance '{}' appears {:d} time(s)."
                   " Yet the list of instances is '{}'.  The list of instances"
                   " should be '{}'."
                   .format(building_metadata['instance'], appliance_type,
                           len(instances), instances, correct_instances))
            raise NilmMetadataError(msg)
