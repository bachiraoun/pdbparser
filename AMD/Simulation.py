#!/Library/Frameworks/Python.framework/Versions/7.0/Resources/Python.app/Contents/MacOS/Python
import importlib
import sys
sys.path.insert(0,"/Users/AOUN/Desktop/pdbparser")

import numpy as np

import Pyro.errors
import Pyro.naming
import subprocess

import DistributedComputing.MasterSlave as MasterSlave

import pdbparser
import SimulationBox
from Utilities import randomVelocity, Array, StructureFileParser


class Simulation(object):
    """
    Main simulation class. It parses the input arguments and get Pyro server ready for
    parallel computing.
    """
    def __init__(self, name, input_files, forcefield_terms, parameters):
        """
        All simulation possible variables are declared herein
        the default values are attributed.
        """
        self.name = name
        # declare simulation variables
        self.simulationBox = None
        self.coordinates = None
        self.velocities = None
        self.structure = None
        self.numberOfAtoms = None
        self.forcefieldTerms = []

        # declare simulation default parameters
        self.parameters = {}
        self.parameters["box_size"] = 10.0
        self.parameters["LennardJones_cutoff"] = 14.0
        self.parameters["temperature"] = 300
        self.parameters.update(parameters)

        # initialize simulation variables
        self.__initialize_input_files_variables__(input_files)

        # initialize forcefield arrays
        self.__initialize_forcefield_terms__(forcefield_terms)


    def __initialize_input_files_variables__(self, input_files):
        # simulation box size
        self.simulationBox = SimulationBox.PeriodicSimulationBox()
        self.simulationBox.setVectors(self.parameters["box_size"])
        # initial coordinates
        pdb = pdbparser.pdbparser(input_files["pdb"])
        self.coordinates = Array(pdb.get_coordinates())
        # number of atoms
        self.numberOfAtoms = self.coordinates.shape[0]
        # structure information
        self.structure = StructureFileParser(input_files["sf"])
        # initialize velocities
        self.velocities = Array( randomVelocity(self.parameters["temperature"], self.structure.atomMass/6.023e23) )


    def __initialize_forcefield_terms__(self, forcefield_terms):
        for term in forcefield_terms.keys():
            try:
                termModule = importlib.import_module("ForceFieldTerms.%s"%term)
                _classInstance = getattr(termModule, term)(self)
            except ImportError:
                raise "Failed to import %s" %term
            _classInstance.__calculation_method__ = _classInstance.methods[forcefield_terms[term]]
            _classInstance.initialize()
            self.forcefieldTerms.append(_classInstance)



    def run(self):
        """
        this is the main simulation run
        """
        for timeIndex in self.parameters["number_of_steps"]:
            self.atomCalculation(index)



    def atomCalculation(self, index):
        """
        calculates the energy and forces of atom index according to forcefield defined terms
        """
        for term in self.forcefieldTerms:
            term()




if __name__ == "__main__":
    forcefield_terms = {"LennardJones":"LennardJones_12_6"}
    input_files = {"pdb": "examples/Argon.pdb", "sf":"examples/Argon.psf"}
    parameters = {"box_size":77.395, "temperature":300, "time_step":1, "number_of_steps":100000}

    simu = Simulation("Argon_simulation", input_files, forcefield_terms, parameters)

    print simu.simulationBox.vectors()
    print simu.coordinates
    print simu.structure
    print simu.velocities
    print simu.forcefieldTerms
    print simu.forcefieldTerms[0].parent
    print simu.forcefieldTerms[0].sigma
