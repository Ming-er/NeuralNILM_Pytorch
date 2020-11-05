#!/usr/bin/env python
from __future__ import print_function
import unittest
from nilm_metadata.convert_yaml_to_hdf5 import (
    _sanity_check_appliances, NilmMetadataError)


class TestConvertYamlToHdf5(unittest.TestCase):

    def test_sanity_check_appliances(self):
        def building(appliances):
            return {
                'appliances': appliances,
                'instance': 1,
                'elec_meters': {1: {}}
            }

        # this should succeed with no errors
        _sanity_check_appliances(building([
            {'instance': 1, 'type': 'fridge freezer', 'meters': [1]}]))

        BAD_APPLIANCES = [
            {'type': 'fridge freezer', 'meters': [1]},  # no instance
            {'meters': [1]},  # no instance or type
            {},  # empty
            {'instance': 1, 'meters': [1]},  # no type
            {'instance': 1, 'type': 'fridge freezer'},  # no meters
            {'instance': 1, 'type': 'fridge freezer',
             'meters': [2]},  # bad meter
            {'instance': 1, 'type': 'blah blah', 'meters': [1]},  # bad type
            {'instance': 2, 'type': 'fridge freezer',
             'meters': [1]},  # bad instance
            ['blah'],  # not a dict
            'blah',  # not a dict
            None  # not a dict
        ]

        for bad_appliance in BAD_APPLIANCES:
            with self.assertRaises(NilmMetadataError):
                _sanity_check_appliances(building([bad_appliance]))


if __name__ == '__main__':
    unittest.main()
