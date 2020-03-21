from __future__ import print_function
import os
import numpy as np

from pdbparser.log import Logger
from pdbparser import pdbparser
from pdbparser.Utilities.Collection import get_path
from pdbparser.Utilities.Selection import NanotubeSelection
from pdbparser.Utilities.Information import get_models_records_indexes_by_records_indexes, get_records_indexes_in_attribute_values
from pdbparser.Utilities.Modify import *
from pdbparser.Utilities.Geometry import get_principal_axis, translate, orient

# read pdb
pdbCNT = pdbparser(os.path.join(get_path("pdbparser"),"Data/nanotubeWaterNAGMA.pdb" ) )

Logger.info("Define models")
# define models
define_models_by_records_attribute_value(pdbCNT.indexes, pdbCNT)

Logger.info("Getting nanotube indexes")
# get CNT indexes
cntIndexes = get_records_indexes_in_attribute_values(pdbCNT.indexes, pdbCNT, "residue_name", "CNT")

Logger.info("Create selection")
# create selection
sel = NanotubeSelection(pdbCNT, nanotubeIndexes = cntIndexes).select()

Logger.info("Get models inside nanotube")
# construct models out of residues
indexes = get_models_records_indexes_by_records_indexes(sel.selections["inside_nanotube"], pdbCNT)

#indexes.extend(sel.selections["nanotube"])
pdb = pdbCNT.get_copy(cntIndexes+indexes)

# orient along X axis
Logger.info("Orient to OX and translate to nanotube center")
center, _, _, _, vect1, _, _ = get_principal_axis(cntIndexes, pdbCNT)
translate( pdb.indexes, pdb, -1.*np.array(center) )
orient(axis=[1,0,0], indexes=pdb.indexes, pdb=pdb, records_axis=vect1)

# delete extra molecules
Logger.info("Delete extra molecules to refine selection and reset models and records serial number")
#delete_records_by_sequence_number(26, pdb)
#delete_records_by_sequence_number(527, pdb)

Logger.info("Reset serial number and sequence identifier")
reset_records_serial_number(pdb)
reset_sequence_identifier_per_record(pdb)

# visualize
pdb.visualize()
