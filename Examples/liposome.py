from __future__ import print_function
import os

from pdbparser.Utilities.Collection import get_path
from pdbparser import pdbparser
from pdbparser.Utilities.Construct import Liposome
from pdbparser.Utilities.Modify import reset_sequence_identifier_per_model

# read SDS molecule from pdbparser database
pdbSDS = pdbparser(os.path.join(get_path("pdbparser"),"Data/SDS.pdb" ) )

# create liposome
pdbLIPOSOME= Liposome(pdbSDS,
                      innerInsertionNumber = 1000,
                      positionsGeneration = "symmetric").construct()

# visualize liposome
pdbLIPOSOME.get_pdb().visualize()
