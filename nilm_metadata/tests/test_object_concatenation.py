#!/usr/bin/env python
from __future__ import print_function
import unittest
from ..object_concatenation import (recursively_update_dict, 
                                    _concatenate_complete_object)

class TestObjectConcatenation(unittest.TestCase):

    def testrecursively_update_dict(self):
        d1 = {}
        d2 = {'a':1, 'b':2, 'c': {'ca':10, 'cb': 20} }
        recursively_update_dict(d1,d2)
        self.assertEqual(d1, d2)

        d1 = {'a':-1, 'b':-3, 'c': {}}
        d2 = {'a':1, 'b':2, 'c': {'ca':10, 'cb': 20} }
        recursively_update_dict(d1,d2)
        self.assertEqual(d1, d2)

        d1 = {'a':-1, 'b':-3, 'c': {}, 'list': [1,2,3]}
        d2 = {'a':1, 'b':2, 'c': {'ca':10, 'cb': 20}, 'list': [4,5,6] }
        recursively_update_dict(d1,d2)
        self.assertEqual(d1, {'a':1, 'b':2, 'c': {'ca':10, 'cb': 20}, 'list': [1,2,3,4,5,6] })

        d1 = {'a':-1, 'b':-3}
        d2 = {'a':1, 'b':2, 'c': {'ca':10, 'cb': 20} }
        recursively_update_dict(d1,d2)
        self.assertEqual(d1, d2)

        d1 = {'a':-1, 'b':-3, 'c': {'ca':-10, 'cc': 30} }
        d2 = {'a':1, 'b':2, 'c': {'ca':10, 'cb': 20} }
        recursively_update_dict(d1,d2)
        self.assertEqual(d1, {'a':1, 'b':2, 'c': {'ca':10, 'cb': 20, 'cc': 30} })

    def test_distance(self):
        objects = {
            "a": {
                "distributions": {
                    "on_power": [
                        {"description": "a"}
                    ],
                    "on_duration": [
                        {"description": "a"}
                    ]
                }
            },
            "b": {
                "parent": "a",
                "distributions": {
                    "on_power": [
                        {"description": "b"}
                    ]
                }
            },
            "c": {
                "parent": "b",
                "distributions": {
                    "on_power": [
                        {"description": "c"}
                    ]
                }
            }
        }
        obj = _concatenate_complete_object('c', objects)
        on_power = obj['distributions']['on_power']
        self.assertEqual(on_power[0], {'distance':2, 'description': 'a',
                                       'from_appliance_type': 'a'})
        self.assertEqual(on_power[1], {'distance':1, 'description': 'b',
                                       'from_appliance_type': 'b'})
        self.assertEqual(on_power[2], {'distance':0, 'description': 'c',
                                       'from_appliance_type': 'c'})
        on_duration = obj['distributions']['on_duration']
        self.assertEqual(on_duration[0], {'distance':2, 'description': 'a',
                                          'from_appliance_type': 'a'})

if __name__ == '__main__':
    unittest.main()
