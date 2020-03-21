# standard libraries imports
from __future__ import print_function
import os, copy
from collections import Counter

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser import pdbparser
from pdbparser.Utilities.Collection import get_path
from pdbparser.Utilities.Database import __ATOM__
from pdbparser.Utilities.Construct import Micelle, Sheet, Nanotube
from pdbparser.Analysis.Structure.SolventAccessibleSurfaceArea import SolventAccessibleSurfaceArea


pdbSDS  = pdbparser(os.path.join(get_path("pdbparser"),"Data/SDS.pdb" ) )
pdbCTAB = pdbparser(os.path.join(get_path("pdbparser"),"Data/CTAB.pdb" ) )

PDB = Nanotube().construct().get_pdb()
PDB = Sheet().construct().get_pdb()
PDB = Micelle([pdbSDS],
               flipPdbs = [True,True],
               positionsGeneration = "symmetric").construct().get_pdb()

SASA = SolventAccessibleSurfaceArea(trajectory=PDB, configurationsIndexes=[0],
                   targetAtomsIndexes=PDB.indexes, atomsRadius='vdwRadius', makeContiguous=False,
                   probeRadius=0, resolution=0.5, storeSurfacePoints=True, tempdir=None)

SASA.run()
print('Surface Accessible Surface Area is:',SASA.results['sasa'][0], 'Ang^2.')

# get points and build fake Cupper surface pdb
points = SASA.results['surface points'][0]

__ATOM__['element_symbol']='cu'
__ATOM__['atom_name']='cu'

surface = pdbparser()
surface.records = [copy.copy(__ATOM__) for _ in range(points.shape[0])]
surface.set_coordinates(points)
PDB.concatenate(surface)

# visualize
PDB.visualize()
