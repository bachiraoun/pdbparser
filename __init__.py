"""
It's a Protein Data Bank (.pdb) files manipulation package that is
mainly developed to parse and load, duplicate, manipulate and create pdb files.
A full description of a pdb file can be found here: http://deposit.rcsb.org/adit/docs/pdb_atom_format.html
pdbparser atoms configuration can be visualized by vmd software (http://www.ks.uiuc.edu/Research/vmd/)
by simply pointing 'VMD_PATH' global variable to the exact path of vmd executable, and using 'visualize' method.
At any time and stage of data manipulation, a pdb file of all atoms or a subset of atoms can be exported to a pdb file.\n
Additional sub-modules (pdbTrajectory, etc) and sub-packages (Analysis, etc)
started to add up to pdbparser package, especially when traditional molecular dynamics data analysis softwares
couldn't keep up the good performance, feasibility and speed of calculation with the increasing
number of atoms in the simulated system.
"""
# get package info
from __future__ import print_function
from .__pkginfo__ import __version__, __author__

# get globals
#from ._globals import get_parameters, write_parameters, set_vmd_path, get_parameter_value, update_parameters

# import important definitions
from .pdbparser import pdbparser, pdbTrajectory
