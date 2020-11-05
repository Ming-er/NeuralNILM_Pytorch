from __future__ import print_function, division
from copy import deepcopy
from .file_management import get_appliance_types_from_disk
from six import iteritems


def get_appliance_types():
    """
    Returns
    -------
    dict of all appliance types.  Fully concatenated and with components
    recursively resolved.
    """
    appliance_types_from_disk = get_appliance_types_from_disk()
    appliance_types = _concatenate_all_appliance_types(
        appliance_types_from_disk)
    return appliance_types


class ObjectConcatenationError(Exception):
    pass


def _concatenate_all_appliance_types(appliance_types_from_disk):
    concatenated = {}
    for appliance_type_name in appliance_types_from_disk:
        concatenated_appliance_type = _concatenate_complete_appliance_type(
            appliance_type_name, appliance_types_from_disk)
        concatenated[appliance_type_name] = concatenated_appliance_type

    return concatenated


def _concatenate_complete_appliance_type(
        appliance_type_name, appliance_types_from_disk):

    concatenated_app_type = _concatenate_complete_object(
        appliance_type_name, appliance_types_from_disk)
    categories = concatenated_app_type.setdefault('categories', {})

    # Instantiate components recursively
    components = concatenated_app_type.get('components', [])
    for i, component_appliance_obj in enumerate(components):
        component_type_name = component_appliance_obj['type']
        component_type_obj = _concatenate_complete_appliance_type(
            component_type_name, appliance_types_from_disk)
        recursively_update_dict(component_appliance_obj, component_type_obj)
        components[i] = component_appliance_obj

        # Now merge component categories into owner appliance type object
        if not component_appliance_obj.get('do_not_merge_categories'):
            recursively_update_dict(
                categories, component_appliance_obj.get('categories', {}))

    return concatenated_app_type


def _init_distributions(appliance_type):
    for list_of_dists in appliance_type.get('distributions', {}).values():
        for dist in list_of_dists:
            dist.update({'from_appliance_type': appliance_type['type'],
                         'distance': 0})


def _concatenate_complete_object(object_name, object_cache):
    """
    Returns
    -------
    merged_object: dict.
        If `child_object` is None then merged_object will be the object
        identified by `object_name` merged with its ancestor tree.
        If `child_object` is not None then it will be merged as the
        most-derived object (i.e. a child of object_name).  This is
        useful for appliances.
    """
    ancestors = _get_ancestors(object_name, object_cache)

    # Now descend from super-object downwards,
    # collecting and updating properties as we go.
    merged_object = deepcopy(ancestors[0])
    _init_distributions(merged_object)
    merged_object['n_ancestors'] = len(ancestors) - 1

    for i, next_child in enumerate(ancestors[1:]):
        # Remove properties that the child does not want to inherit
        do_not_inherit = next_child.get('do_not_inherit', [])
        do_not_inherit.extend(['synonyms', 'description', 'do_not_inherit'])
        for property_to_not_inherit in do_not_inherit:
            merged_object.pop(property_to_not_inherit, None)

        # Now, for each probability distribution, we tag it with a
        # 'distance' property, showing how far away it is from
        # the most derived object.
        distributions = merged_object.get('distributions', {})
        for list_of_dists in distributions.values():
            for dist in list_of_dists:
                dist['distance'] += 1

        _init_distributions(next_child)

        recursively_update_dict(merged_object, next_child)

    return merged_object


def _get_ancestors(appliance_type_name, appliance_types_from_disk):
    """
    Arguments
    ---------
    appliance_type_name: string

    Returns
    -------
    A list of dicts where each dict is an object. The first
    dict is the highest on the inheritance hierarchy; the last dict
    is the object with type == `appliance_type_name`.

    Raises
    ------
    ObjectConcatenationError
    """
    if appliance_type_name is None:
        return []

    # walk the inheritance tree from
    # bottom upwards (which is the wrong direction
    # for actually doing inheritance)
    try:
        current_appliance_type_dict = appliance_types_from_disk[
            appliance_type_name]
    except KeyError as e:
        msg = "'{}' not found!".format(appliance_type_name)
        raise ObjectConcatenationError(msg)

    current_appliance_type_dict['type'] = appliance_type_name
    ancestors = [current_appliance_type_dict]

    while current_appliance_type_dict.get('parent'):
        parent_type = current_appliance_type_dict['parent']
        try:
            current_appliance_type_dict = appliance_types_from_disk[
                parent_type]
        except KeyError as e:
            msg = ("Object '{}' claims its parent is '{}' but that"
                   " object is not recognised!"
                   .format(current_appliance_type_dict['type'], e))
            raise ObjectConcatenationError(msg)

        current_appliance_type_dict['type'] = parent_type
        ancestors.append(current_appliance_type_dict)

    ancestors.reverse()
    return ancestors


def recursively_update_dict(dict_to_update, source_dict):
    """ Recursively extends lists in dict_to_update with lists in source_dict,
    and updates dicts.

    This function is required because Python's `dict.update()` function
    does not descend into dicts within dicts.

    Parameters
    ----------
    dict_to_update, source_dict : dict
        Updates `dict_to_update` in place.
    """
    source_dict = deepcopy(source_dict)
    for key_from_source, value_from_source in iteritems(source_dict):
        try:
            value_to_update = dict_to_update[key_from_source]
        except KeyError:
            dict_to_update[key_from_source] = value_from_source
        else:
            if isinstance(value_from_source, dict):
                assert isinstance(value_to_update, dict)
                recursively_update_dict(value_to_update, value_from_source)
            elif isinstance(value_from_source, list):
                assert isinstance(value_to_update, list)
                value_to_update.extend(value_from_source)
                if not any([isinstance(v, dict) for v in value_to_update]):
                    dict_to_update[key_from_source] = list(
                        set(value_to_update))
            else:
                dict_to_update[key_from_source] = value_from_source
