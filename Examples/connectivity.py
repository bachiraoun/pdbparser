"""
Construct a graphene sheet in two orientations
"""
from __future__ import print_function
import os

from pdbparser.Utilities.Collection import get_path
from pdbparser.log import Logger
from pdbparser import pdbparser
from pdbparser.Utilities.Connectivity import Connectivity
from pdbparser.Utilities.Modify import reset_atom_name

Logger.info("reading SDBS molecule")
pdb = pdbparser(os.path.join(get_path("pdbparser"),"Data/connectivityTestMolecule.pdb" ) )

connectivity = Connectivity(pdb)

# bonds
Logger.info("calculating bonds")
connectivity.calculate_bonds()
bonds = connectivity.get_bonds(key = "atom_name")
print("BONDS:")
print("======")
for idx in range(len(bonds[0])):
    print("%6s"%bonds[0][idx], ' = ', bonds[1][idx])
#connectivity.export_bonds('lol.psf')

# angles
Logger.info("calculating angles")
connectivity.calculate_angles()
angles = connectivity.get_angles(key = "atom_name")
print("ANGLES:")
print("======")
for angle in angles:
    print(angle)
#connectivity.export_angles('lol.psf')

# dihedrals
Logger.info("calculating dihedrals")
connectivity.calculate_dihedrals()
dihedrals = connectivity.get_dihedrals(key = "atom_name")
print("DIHEDRALS:")
print("=========")
for dihedral in dihedrals:
    print(dihedral)
#connectivity.export_dihedrals('lol.psf')


print(connectivity.get_dihedrals())
#pdb.visualize()
