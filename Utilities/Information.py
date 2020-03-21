"""
This module contains all methods used to extract information from pdbparser instance.
"""
# standard libraries imports
from __future__ import print_function

# external libraries imports
import numpy as np

# pdbparser library imports
from ..log import Logger
import pdbparser.Utilities.Database as DB


def get_records_indexes_by_attribute_value(indexes, pdb, attribute, value):
    """
        Get all records verifying pdb.records[attribute] = value.\n
        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the indexes of pdb.
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. attribute (string): record attribute name.
            #. value (object): the desired value.

        :Returns:
            #. indexes (list): all found records indexes.
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    return [idx for idx in indexes if pdb.records[idx][attribute] == value]


def get_records_indexes_in_attribute_values(indexes, pdb, attribute, values):
    """
        Get all records verifying pdb.records[attribute] in values.\n
        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the indexes of pdb.
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. attribute (string): record attribute name.
            #. values (list): list of desired values.

        :Returns:
            #. indexes (list): all found records indexes.
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    return [index for index in indexes if pdb.records[index][attribute] in values]


def get_records_attribute_values(indexes, pdb, attribute):
    """
        Get all records attributes values.\n
        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the indexes of pdb.
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. attribute (string): record attribute name.

        :Returns:
            #. values (list): all found records values.
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    return [pdb.records[idx][attribute] for idx in indexes]

def get_number_of_residues(indexes, pdb):
    """
        Calculate the number of every residue type in pdb file.\n
        residue_name, sequence_number and segment_identifier attributes in pdb file must be correct.

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the indexes of pdb.
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.

        :Returns:
            #. residues (dictionary): keys all residues and values are number encountered.
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    res = get_records_attribute_values(indexes, pdb, "residue_name")
    seq = get_records_attribute_values(indexes, pdb, "sequence_number")
    sid = get_records_attribute_values(indexes, pdb, "segment_identifier")
    # create residues dict
    residues = dict(zip(set(res), [0]*len(set(res))))
    currentSeq = False
    currentSid = False
    for idx in range(len(seq)):
        if seq[idx] != currentSeq or sid[idx] != currentSid:
            currentSeq = seq[idx]
            currentSid = sid[idx]
            residues[res[idx]] += 1
    return residues

def get_records_database_property_values(indexes, pdb, property):
    """
        Return records database property values

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the indexes of pdb.
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. property (string): the property in pdbparser database

        :Returns:
            #. values (list): records property values.
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    elements = [pdb.records[idx]["element_symbol"] for idx in indexes]
    return [DB.__atoms_database__[el.strip().lower()][property] for el in elements]


def get_coordinates(indexes, pdb):
    """
        Return records coordinates numpy.array of shape (numberOfRecords, 3)

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the indexes of pdb.
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.

        :Returns:
            #. coordinates (numpy.array): records coordinates
    """
    if pdb.__class__.__name__ == "pdbparser":
        records = pdb.records
        X = [records[idx]['coordinates_x']  for idx in indexes]
        Y = [records[idx]['coordinates_y']  for idx in indexes]
        Z = [records[idx]['coordinates_z']  for idx in indexes]
        #return X, Y, Z
        return np.transpose([X, Y, Z])
    elif pdb.__class__.__name__ == "pdbTrajectory":
        return pdb.coordinates[indexes,:]
    else:
        raise Logger.error("pdb must be a pdbparser or pdbTrajectory instance")


def get_records_indexes_by_models_keys(pdb, keys):
    """
        Return records indexes defined in model

        :Parameters:
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. keys (list, tuple, set): list of keys

        :Returns:
            #. indexes (list): the found indexes .
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    indexes = []
    keys = list(set(keys))
    for key in keys:
        indexes.extend(range(pdb.models[key]["model_start"], pdb.models[key]["model_end"]))
    return indexes


def get_models_records_indexes_by_records_indexes(indexes, pdb):
    """
        Returns all the records indexes sharing the same model with any record in indexes

        :Parameters:
            #. indexes (list, tuple, numpy.ndarray): the indexes of pdb.
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.

        :Returns:
            #. indexes (list): the found indexes .
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    modelsKeys = set([get_model_key_by_record_index(index=idx, pdb=pdb) for idx in indexes])
    return get_records_indexes_by_models_keys(keys=modelsKeys, pdb=pdb)


def get_models_keys_by_attribute_value(pdb, keys, attribute, value):
    """
        Returns all models keys having pdb.models[key][attribute] = value

        :Parameters:
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. keys (list, tuple, set): list of keys
            #. attribute (string): record attribute name.
            #. value (object): the desired value

        :Returns:
            #. indexes (list): the found indexes .
     """
    if pdb.__class__.__name__ == "pdbTrajectory":
       pdb = pdb._structure
    else:
       assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    return [key for key in keys if pdb.models[key][attribute] == value]


def get_models_attribute_values(pdb, keys, attribute):
    """
        Returns all pdb.models[key][attribute] values

        :Parameters:
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. key (integer): the model key
            #. attribute (string): record attribute name.

        :Returns:
            #. values (list): the list of values.
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    return [pdb.models[key][attribute] for key in keys]


def get_model_key_by_record_index(pdb, index):
    """
        return the model key if index is in model range. Otherwise None

        :Parameters:
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. index (integer): the records index

        :Returns:
            #. key (integer): the found model key.
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    models = pdb.models
    for key in models.keys():
        if (models[key]['model_start'] <= index) and (models[key]['model_end'] > index):
            return key
    return None


def get_model_key_by_model_serial_number(pdb, serialNumber):
    """
        return the model key having the same given serialNumber

        :Parameters:
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. serialNumber (integer): the serial number.

        :Returns:
            #. key (integer): the found model key.
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    models = pdb.models
    for key in models.keys():
        if models[key]['model_serial_number'] == serialNumber:
            return key

    return None


def get_model_key_by_record_serial_number(pdb, serialNumber):
    """
        return the model key of the first record matching serialNumber

        :Parameters:
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. serialNumber (integer): the serial number.

        :Returns:
            #. key (integer): the found model key.
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    indexes = get_records_indexes_by_attribute_value(pdb.indexes, pdb, "serial_number", serialNumber)
    if not len(indexes):
        return None
    else:
        return get_model_key_by_record_index(index=indexes[0], pdb=pdb)



def get_model_key_by_record_sequence_number(pdb, sequenceNumber):
    """
        return the model key of the first record matching sequenceNumber.

        :Parameters:
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. sequenceNumber (integer): the serial number.

        :Returns:
            #. key (integer): the found model key.
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    indexes = get_records_indexes_by_attribute_value(pdb.indexes, pdb, "sequence_number", sequenceNumber)
    if not len(indexes):
        return None
    else:
        return get_model_key_by_record_index(index=indexes[0], pdb=pdb)



def get_model_attribute_value_by_record_index( pdb, index, attribute):
    """
        Returns the model attribute value if records index is in model range,

        :Parameters:
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. index (integer): the record index
            #. attribute (integer): the serial number.

        :Returns:
            #. value (object): the model attribute value
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    models = pdb.models
    for key in models.keys():
        if (models[key]['model_start'] <= index) and (models[key]['model_end'] > index):
            return models[key][attribute]

    return None


def get_model_range_by_attribute_value(pdb, keys, attribute, value):
    """
        Returns (pdb.models[key]["model_start"], pdb.models[key]["model_end"])
        at the first found pdb.models[key][attribute] = value

        :Parameters:
            #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
            #. keys (list, tuple, set): list of keys
            #. attribute (string): record attribute name.
            #. value (object): the desired value

        :Returns:
            #. range (tuple): indexes range of model
    """
    if pdb.__class__.__name__ == "pdbTrajectory":
        pdb = pdb._structure
    else:
        assert pdb.__class__.__name__ == "pdbparser",  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    for key in keys:
        if pdb.models[key][attribute] == value:
            return (pdb.models[key]["model_start"], pdb.models[key]["model_end"])

    return None


def get_trajectory_indexes(pdb, indexes):
    """
    check and return indexes if they are in trajectory's range.\n

    :Parameters:
        #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
        #. indexes (list): The list of indexes

    :Returns:
        #. indexes (list): the verified list of indexes
    """

    assert pdb.__class__.__name__ in ("pdbparser", "pdbTrajectory"),  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    assert isinstance(indexes, (list, set, tuple)), Logger.error("indexes must be a list of positive integers smaller than trajectory's length")
    indexes = sorted(set(indexes))
    nConf = pdb.numberOfConfigurations
    assert not len([False for idx in indexes if (idx%1!=0 or idx<0 or idx>=nConf)]), Logger.error("indexes must be a list of positive integers smaller than trajectory's length")
    return [int(idx) for idx in set(indexes)]


def get_atoms_indexes(pdb, indexes):
    """
    check and return indexes if they are in trajectory number of atoms range.\n

    :Parameters:
        #. pdb (pdbparser, pdbTrajectory): the pdbparser of pdbTrajectory instance.
        #. indexes (list): The list of indexes

    :Returns:
        #. indexes (list): the verified list of indexes
    """
    assert pdb.__class__.__name__ in ("pdbparser", "pdbTrajectory"),  Logger.error("pdb must be pdbparser or pdbTrajectory instance")
    assert isinstance(indexes, (list, set, tuple)), Logger.error("indexes must be a list of positive integers smaller than number of atoms")
    indexes = sorted(set(indexes))
    nAtoms = pdb.numberOfAtoms
    assert not len([False for idx in indexes if (idx%1!=0 or idx<0 or idx>=nAtoms)]), Logger.error("indexes must be a list of positive integers smaller than number of atoms")
    return [int(idx) for idx in set(indexes)]
