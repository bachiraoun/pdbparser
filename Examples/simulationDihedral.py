from __future__ import print_function
import os
import tempfile
import copy

import numpy as np

from pdbparser import pdbparser
from pdbparser.log import Logger
from pdbparser.Utilities.Database import __ATOM__
from pdbparser.Utilities.Collection import get_path
from pdbparser.Utilities.Simulate import Simulation

at1 = copy.deepcopy(__ATOM__)
at2 = copy.deepcopy(__ATOM__)
at3 = copy.deepcopy(__ATOM__)
at4 = copy.deepcopy(__ATOM__)
at5 = copy.deepcopy(__ATOM__)
at6 = copy.deepcopy(__ATOM__)
at7 = copy.deepcopy(__ATOM__)
at8 = copy.deepcopy(__ATOM__)
at1['atom_name'] = "c1"
at2['atom_name'] = "c2"
at3['atom_name'] = "c3"
at4['atom_name'] = "c4"
at5['atom_name'] = "c5"
at6['atom_name'] = "c6"
at7['atom_name'] = "c7"
at8['atom_name'] = "c8"
at1['residue_name'] = "c"
at2['residue_name'] = "c"
at3['residue_name'] = "c"
at4['residue_name'] = "c"
at5['residue_name'] = "c"
at6['residue_name'] = "c"
at7['residue_name'] = "c"
at8['residue_name'] = "c"
at1['element_symbol'] = "c"
at2['element_symbol'] = "c"
at3['element_symbol'] = "c"
at4['element_symbol'] = "c"
at5['element_symbol'] = "c"
at6['element_symbol'] = "c"
at7['element_symbol'] = "c"
at8['element_symbol'] = "c"
at1['coordinates_x'] =  0.579980
at2['coordinates_x'] =  1.693270
at3['coordinates_x'] =  2.806730
at4['coordinates_x'] =  3.920020
at5['coordinates_x'] =  5.033480
at6['coordinates_x'] =  6.146940
at7['coordinates_x'] =  7.260400
at8['coordinates_x'] =  8.373860
at1['coordinates_y'] =  0.368530
at2['coordinates_y'] = -0.368280
at3['coordinates_y'] =  0.368280
at4['coordinates_y'] = -0.368530
at5['coordinates_y'] =  0.368530
at6['coordinates_y'] = -0.368280
at7['coordinates_y'] =  0.368280
at8['coordinates_y'] = -0.368530

# create molecule molecule
pdb = pdbparser()
pdb.records = [at1, at2, at3, at4]#, at5, at6, at7, at8]
#pdb.visualize()

# create simulation
sim = Simulation(pdb, logStatus = True, logExport = False,
                 numberOfSteps = 100, outputFrequency = 1,
                 exportInitialConfiguration = True, outputPath = tempfile.mktemp(".xyz"))


# remove all bonded and non bonded interactions except dihedrals
sim.bonds_indexes = []
sim.angles_indexes = []
sim.lennardJones_eps *= 0
sim.atomsCharge *= 0

# initial parameters
sim.__DIHEDRAL__["c c c c"] = {1.0: {'delta': 40.0, 'n': 1.0, 'kchi': 3.6375696}}
sim.set_dihedrals_parameters()
Logger.info("minimizing %s steps at %s fm per step, with all terms suppressed but dihedral %s" % (sim.numberOfSteps, sim.timeStep, sim.__DIHEDRAL__["c c c c"]) )
sim.minimize_steepest_descent()

sim.exportInitialConfiguration = False

# initial parameters
sim.__DIHEDRAL__["c c c c"] = {1.0: {'delta': 120.0, 'n': 1.0, 'kchi': 3.6375696}}
sim.set_dihedrals_parameters()
Logger.info("minimizing %s steps at %s fm per step, with all terms suppressed but dihedral %s" % (sim.numberOfSteps, sim.timeStep, sim.__DIHEDRAL__["c c c c"]) )
sim.minimize_steepest_descent()

# sim.__DIHEDRAL__["c c c c"] = {1.0: {'delta': 0.0, 'n': 1.0, 'kchi': 3.6375696}, 2.0: {'delta': 180.0, 'n': 2.0, 'kchi': -0.328444}, 3.0: {'delta': 0.0, 'n': 3.0, 'kchi': 0.5832496}}
# sim.set_dihedrals_parameters()
# Logger.info("minimizing %s steps at %s fm per step, with all terms suppressed but dihedral %s" % (sim.numberOfSteps, sim.stepTime, sim.__DIHEDRAL__["c c c c"]) )
# sim.minimize_steepest_descent(200)


# visualize molecule
sim.visualize_trajectory(sim.outputPath)
