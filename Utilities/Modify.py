"""
This module contains method that will change the pdbparser atoms and records values.
"""
# standard libraries imports
from __future__ import print_function

# external libraries imports
import numpy as np

# pdbparser library imports
from ..log import Logger
from .Information import *
from .Geometry import *


def set_records_attribute_value(indexes, pdb, attribute, value):
    """
    changes all records in indexes attribute value
    """
    for idx in indexes:
        pdb.records[idx][attribute] = value


def set_records_attribute_values(indexes, pdb, attribute, values):
    """
    changes all records in indexes attribute value
    """
    for idx in indexes:
        pdb.records[idx][attribute] = values[idx]


def reset_atom_name(indexes, pdb):
    """
    Automatically modifies atom_name attibute for every atom in records,
    and makes it unique among all atoms
    """
    # get atomNames strip
    atomNames = [name.strip() for name in get_records_attribute_values(indexes, pdb, "atom_name")]

    # mapping names [(name1,[indexes1]) , (name2,[indexes2]) , ...]
    occurrences = lambda s, lst: (i for i,e in enumerate(lst) if e == s)
    mapp = [ [atomNames[idx], list(occurrences(atomNames[idx], atomNames))] for idx in range(len(atomNames)) if atomNames[idx] not in atomNames[0:idx] ]

    # get atomNames unique
    for lst in mapp:
        # name is unique
        if len(lst[1]) == 1:
            continue
        # name is redundant
        num = 1
        for idx in lst[1]:
            pdb.records[idx]['atom_name'] = pdb.records[idx]['atom_name'].strip() + str(num)
            num = num+1


def set_residue_name(indexes, pdb, name):
    """
    changes all records in indexes residue_name
    """
    for idx in indexes:
        pdb.records[idx]['residue_name'] = name


def set_coordinates(indexes, pdb, coordinates):
    """
    set records coordinates.
    coordinates is an array of shape (len(indexes),3)
    """
    #print(coordinates)
    for idx in range(len(indexes)):
        pdb.records[indexes[idx]]['coordinates_x'] = coordinates[idx,0]
        pdb.records[indexes[idx]]['coordinates_y'] = coordinates[idx,1]
        pdb.records[indexes[idx]]['coordinates_z'] = coordinates[idx,2]


def change_atom_name(name, pdb, newName):
    """
    changes all records of atom_name = name to newName
    """
    indexes = get_records_indexes_by_attribute_value(range(len(pdb.records)), pdb, "atom_name", name)
    for idx in indexes:
        pdb.records[idx]['atom_name'] = newName


def reset_records_serial_number(pdb, start = 1):
    """
    this method simply reset all records 'model_serial_number' attribute
    if models is True, it resets also the records serial numbers at every model start
    """
    SN = int(start)
    for idx in pdb.indexes:
        pdb.records[idx]['serial_number'] = SN
        if idx in pdb.anisou:
            pdb.anisou[idx]['serial_number'] = SN
        SN += 1

def reset_sequence_identifier_per_model(keys, pdb, start = 1):
    """
    Automatically reset sequence_identifier attibute for every atom in records,
    sequence_number incremented for every model
    makes it serial starting the account according to 'start' attibute
    default: start = 1
    """
    SI = int(start)
    for key in keys:
        modelIndexes = range(pdb.models[key]["model_start"],pdb.models[key]["model_end"])
        for idx in modelIndexes:
            pdb.records[idx]["sequence_number"] = SI
        SI += 1


def reset_sequence_identifier_per_record(pdb, start = 1):
    """
    Automatically reset sequence_identifier attibute for every atom in records,
    makes it serial starting the account according to 'start' attibute
    sequence_number incremented at every change of the original value
    default: start = 1
    """
    SI = int(start)
    last = pdb.records[0]["sequence_number"]
    for idx in pdb.indexes:
        if pdb.records[idx]["sequence_number"] == last:
            pdb.records[idx]["sequence_number"] = SI
        else:
            SI += 1
            last = pdb.records[idx]["sequence_number"]
            pdb.records[idx]["sequence_number"] = SI


def reset_sequence_number_per_residue(indexes, pdb, name = None, start = 1):
    """
    Automatically reset sequence_identifier attibute for every atom in records,
    makes it serial starting the account according to 'start' attibute
    sequence_number incremented at every change in the residue name
    default: start = 1
    if start = None, it takes the value of maximum sequence_number attribute found +1
    """
    # get indexes in case Name is not None
    if name is not None:
        indexes = [idx for idx in indexes  if pdb.records[idx]["residue_name"] == name]

    if len(indexes) == 0:
        return

    # get the maximum sequence number found
    maxSN = np.max([pdb.records[idx]["sequence_number"] for idx in indexes])

    # get start
    if start is None:
        start = maxSN + 1

    SN = start
    lastName = pdb.records[indexes[0]]["residue_name"]
    lastSN= pdb.records[indexes[0]]["sequence_number"]
    for idx in indexes:
        if (pdb.records[idx]["residue_name"] == lastName) and (pdb.records[idx]["sequence_number"] == lastSN):
            pdb.records[idx]["sequence_number"] = SN
        else:
            SN += 1
            lastName = pdb.records[idx]["residue_name"]
            lastSN= pdb.records[idx]["sequence_number"]
            pdb.records[idx]["sequence_number"] = SN

    reset_models_ter_sequence_number(list(pdb.models.keys()), pdb)


def reset_segments(indexes, pdb, basename = None, start = 0, segment_size = 9999):
    """
    Automatically reset sequence_identifier attibute after 9999 residue,
    automatically set "identifier" name using basename parameter
    """
    assert isinstance(start, int)
    assert start>=0
    assert isinstance(segment_size, int)
    assert segment_size>0

    if basename is None:
        basename = ""

    for idx in indexes:
        segmentNumber = int( (pdb.records[idx]["sequence_number"]-1)/segment_size )
        pdb.records[idx]["sequence_number"] = (pdb.records[idx]["sequence_number"]-1)%segment_size +1
        pdb.records[idx]["segment_identifier"] =  basename + "%s" %(segmentNumber+start)


def reset_models_names(pdb, basename = "model "):
    """
    redefines all models names automatically using a basename
    """
    for key in list(pdb.models.keys()):
        pdb.models[key]['MODEL_NAME'] = basename + str(key)


def reset_models_serial_number(pdb, start = 1):
    """
    this method simply reset all models 'model_serial_number' attribute
    """
    newModels = {}
    SN = start
    for model in pdb.models.values():
        newModels[SN] = model
        newModels[SN]['model_serial_number'] = SN
        SN += 1

    pdb.models = newModels


def set_sequence_number(indexes, pdb, number):
    """
    changes all records of sequence_number attribute to number
    """
    for index in indexes:
        pdb.records[index]["sequence_number"] = number


def increment_sequence_number(indexes, pdb, increment = 1):
    """
    increment all records of sequence_number attribute
    """
    for index in indexes:
        pdb.records[index]["sequence_number"] += increment

    reset_models_ter_sequence_number(pdb.models.keys(), pdb)


def reset_models_ter_sequence_number(keys, pdb):
    """
    set the models ter sequence number equal to the final record sequence number in the model
    """
    for key in keys:
        pdb.models[key]['termodel']['sequence_number'] = pdb.records[pdb.models[key]['model_end']-1]['sequence_number']



def define_models_by_records_attribute_value(indexes, pdb, attribute = "sequence_number", reset = False):
    """
    Defines models by looking at records attribute value
    if reset is True, all exisiting models will be deleted
    """
    if reset:
        pdb.delete_all_models_definition()

    modelStart = indexes[0]
    attributeValue = pdb.records[0][attribute]
    for idx in indexes:
        attVal = pdb.records[idx][attribute]
        if attributeValue != attVal:
            pdb.define_model(model_start = modelStart, model_end = idx, model_name = None)
            attributeValue = attVal
            modelStart = idx

    # check for last model
    if (modelStart != indexes[-1]) and (attributeValue == attVal):
        pdb.define_model(model_start = modelStart, model_end = indexes[-1]+1, model_name = None)


def change_model_name(name, newName, pdb):
    """
    change MODEL_NAME attribute
    """
    keys = pdb.get_model_key_by_name(name)
    for key in keys:
        pdb.models[key]["MODEL_NAME"] = newName



def delete_model_definition_by_name(name, pdb):
    """
    deletes ontly the model definition using its name but not its associated records
    """
    key = pdb.get_model_key_by_name(name)
    if key is not None:
        pdb.models.pop(key)


def delete_records_of_model_by_name(name, pdb):
    """
    delete the records and the model definition using the model name
    """
    keys = pdb.get_model_key_by_name(pdb)
    for key in keys:
        model = pdb.models[key]
        delete_records(range(model["model_start"],model["model_end"]), pdb)


def delete_records_of_model_by_index(index, pdb):
    """
    deletes the records and the model definition using the record index
    """
    key = get_model_key_by_record_index(index, pdb)
    if key is not None:
        model = pdb.models[key]
        delete_records(range(model["model_start"],model["model_end"]), pdb)


def delete_records_by_model_serial_number(serial_number, pdb):
    """
    deletes the records and the model definition using the record index
    """
    key = get_model_key_by_model_serial_number(serial_number, pdb)
    if key is not None:
        model = pdb.models[key]
        delete_records(range(model["model_start"],model["model_end"]), pdb)


def delete_records_by_sequence_number(sequence_number, pdb):
    """
    deletes the records and the model definition using the record index
    """
    indexes = get_records_indexes_by_attribute_value(pdb.indexes, pdb, "sequence_number", sequence_number)
    delete_records(indexes, pdb)



def delete_records_and_models_records(indexes, pdb):
    """
    Find all records and their associated models records if exist and delete them
    """
    keys = [get_model_key_by_record_index(pdb, idx) for idx in indexes]
    recordsIndexes = sorted( set([indexes[idx] for idx in range(len(keys)) if keys[idx] is None]) )
    keys = filter(None, set(keys) )
    for key in keys:
        recordsIndexes.extend( range(pdb.models[key]["model_start"],pdb.models[key]["model_end"]) )

    delete_records(recordsIndexes, pdb)


def delete_records(indexes, pdb):
    """
    deleting atoms from records can generate errors in the models definition.
    This method is meant to assure the models definitions are correct.
    """
    # to keep records indexes
    indexes = set([idx for idx in range(len(pdb.records)) if idx not in indexes])
    indexes = sorted(indexes)

    # check and pop models
    for key in list(pdb.models.keys()):
        modelRange = set( range( pdb.models[key]["model_start"], pdb.models[key]["model_end"] ) )
        if set.intersection(modelRange, indexes) != modelRange:
            pdb.models.pop(key)

    # correct models range
    for model in pdb.models.values():
        model_start = model["model_start"]
        model_range = model["model_end"] - model_start
        model["model_start"] = indexes.index(model_start)
        model["model_end"] = model["model_start"]+model_range
        model["termodel"]["INDEX_IN_RECORDS"] = model["model_end"]

    # check and pop or correct pdb.ter
    for key in pdb.ter.keys():
        if pdb.ter[key]["INDEX_IN_RECORDS"] not in indexes:
            pdb.ter.pop(key)
        else:
            ter = copy.deepcopy(pdb.ter[key])
            pdb.ter.pop(key)
            newKey = indexes.index(ter["INDEX_IN_RECORDS"])
            ter["INDEX_IN_RECORDS"] = newKey
            pdb.ter[newKey] = ter

    # reset serial numbers
    reset_models_serial_number(pdb)

    # delete records
    pdb.records = [pdb.records[idx] for idx in indexes]


def shake_models(pdb, models_keys = None, threshold = None, intensity_ratio = None):
    """
    This uses a monte-carlo based algorithm to shake models positions
    pdb is a pdbparser instance
    models_keys is the list of models keys that need to be shaken
    threshold is the intermolecular distance threshold
    intensity_ratio is the shaking intensity ratio.
    """
    if models_keys is None:
        models_keys = list(pdb.models.keys())
    numberOfModels = len(models_keys)

    if numberOfModels<=1:
        Logger.info("more than one model should be defined")
        return

    if threshold is None:
        from Utilities.Database import __interMolecularMinimumDistance__
        threshold = __interMolecularMinimumDistance__

    Logger.info("Shaking the models with a minimum inter-models distances threshold of %s with an intensity ratio %s"%(threshold, intensity_ratio))

    # get shaking parameters
    signs = np.sign( np.random.rand(numberOfModels,3)-0.5 )
    shakingDirection = np.random.rand(numberOfModels,3) * signs

    if intensity_ratio is None:
        shakingRatio = np.random.rand(numberOfModels)
    elif np.abs(intensity_ratio) > 1:
        shakingRatio = np.sign( np.ones(numberOfModels) )
    else:
        shakingRatio = intensity_ratio * np.ones(numberOfModels)

    # get all records coordinates
    recordsCoords = np.ma.array( np.transpose( get_coordinates(range(len(pdb.records)), pdb.records) ) )

    for idx in range(numberOfModels):
        model = pdb.models[models_keys[idx]]
        modelRecordsIndexes = range(model["model_start"],model["model_end"])

        # get all model records coordinates
        modelsRecordsCoords = recordsCoords[modelRecordsIndexes]

        # create mask
        recordsCoords.mask = False
        recordsCoords[modelRecordsIndexes] = np.ma.masked

        # calculate minimum inter-models distance
        minPositiveDistance = []
        for modelCoords in modelsRecordsCoords:
            differences = recordsCoords[~recordsCoords.mask].reshape([-1,3]) - modelCoords
            distances = np.sqrt( np.add.reduce(differences*differences, 1) )
            minPositiveDistance.append( np.min( distances ) )
        minPositiveDistance = np.min(minPositiveDistance)

        # get maximum acceptable shaking distance in comparison to the given threshold
        if minPositiveDistance <= threshold:
            continue
        maximumShakingDistance = minPositiveDistance - threshold

        # weight shaking distance
        shakingDistance = shakingRatio[idx] * maximumShakingDistance

        # get normalized direction and calulate shaking vector
        direction =  shakingDirection[idx]/np.linalg.norm( shakingDirection[idx] )
        shakingVector = direction * shakingDistance

        # shake records coords
        translate(modelRecordsIndexes, pdb.records, shakingVector)
