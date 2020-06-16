# standard libraries imports
from __future__ import print_function
import os
import sys
import copy

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser import pdbparser
from pdbparser.log import Logger
from .Geometry import *
from .Information import *
from .Modify import *
from .Connectivity import Connectivity
from .Database import __avogadroNumber__, __coulombConstant__, __boltzmannConstant__
from pdbparser.Utilities.BoundaryConditions import InfiniteBoundaries, PeriodicBoundaries

# conversions from SI(kg,m,s) to (g,A,fs)
__convert_kj_mol__ = 10**-04 # convert kilo joules /mol ---> g/mol A^2 fs^-2
__convert_kn_mol__ = 10**-14 # convert kilo newtons /mol ---> g/mol A^1 fs^-2
__convert_j__      = 10**-3*__convert_kj_mol__*__avogadroNumber__ # convert joules ---> g/mol A^2 fs^-2
__convert_pa__     = __avogadroNumber__*10**3/10**10/10**30 # convert pa=kg/m/s^2 ---> g/mol /A/fs2

# all energy terms must be expressed in kj/mol
# distances in Angstrom and angles in degrees

# V(Lennard-Jones) = Eps,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6] #
# epsilon: kj/mole, Eps,i,j = sqrt(eps,i * eps,j)                       #
# Rmin/2: A, Rmin,i,j = Rmin/2,i + Rmin/2,j                             #
LJ = {}
LJ["c"] = {"eps":0.110000 * 4.184, "rmin/2":2.000000}
LJ["o"] = {"eps":0.120000 * 4.184, "rmin/2":1.700000}
LJ["n"] = {"eps":0.200000 * 4.184, "rmin/2":1.850000}
LJ["h"] = {"eps":0.046000 * 4.184, "rmin/2":0.224500}
LJ["ar"] = {"eps":0.2 * 4.184, "rmin/2":1.7025}
LJ["cl"] = {"eps":0.2 * 4.184, "rmin/2":1.7025}

# V(bond) = Kb(b - b0)**2 #
# Kb: kj/mole/A**2        #
# b0: A                   #
BOND = {}
BOND["c c"] = {"kb":600.000 * 4.184  , "b0":1.3350}
BOND["c h"] = {"kb":300.000 * 4.184  , "b0":1.1000}
BOND["c n"] = {"kb":260.000 * 4.184  , "b0":1.3000}
BOND["h o"] = {"kb":450.000 * 4.184  , "b0":0.9572}
BOND["h h"] = {"kb":400.000 * 4.184  , "b0":0.7414} # hydrogen of h-h molecule

# V(angle) = Ktheta(Theta - Theta0)**2 #
# Ktheta: kj/mole/rad**2               #
# Theta0: degrees                      #
ANGLE = {}
ANGLE["c c c"] = {"ktheta":58.000 * 4.184  , "theta0":113.000}
ANGLE["c c h"] = {"ktheta":26.500 * 4.184  , "theta0":110.100}
ANGLE["c c n"] = {"ktheta":67.700 * 4.184  , "theta0":115.000}
ANGLE["c n c"] = {"ktheta":60.000 * 4.184  , "theta0":109.500}
ANGLE["h c h"] = {"ktheta":35.500 * 4.184  , "theta0":109.000}
ANGLE["h c n"] = {"ktheta":40.000 * 4.184  , "theta0":109.000}
ANGLE["n c n"] = {"ktheta":60.000 * 4.184  , "theta0":109.500}
ANGLE["h o h"] = {"ktheta":55.000 * 4.184  , "theta0":104.520}

# V(dihedral) = Kchi(1 + cos(n(chi) - delta)) #
# Kchi: kj/mole                               #
# n: multiplicity                             #
# delta: degrees                              #
DIHEDRAL = {}
DIHEDRAL["c c c c"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c c c h"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c c c n"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c c h c"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c c h h"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c c h n"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c c n c"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c c n h"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c c n n"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c h c c"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c h c h"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c h c n"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c n c c"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c n c h"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["c n c n"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["h c c h"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["h c c n"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}
DIHEDRAL["n c c n"] = {1: {"n":1, "kchi":0.000 * 4.184  , "delta":0.000}}

IMPROPER = {}



def convert_namd_top_file(top, output):
    """
    """
    # open file and get lines
    fd = open(top,'r')
    lines = fd.readlines()
    fd.close()

    # initialize flag
    resname = None
    residues = {}
    typesMass = {}

    for line in lines:
        splited = line.strip().split()
        # empty line
        if not splited:
            continue
        # header
        if splited[0][0] == "*":
            continue
        # comment line
        if splited[0][0] == "!":
            continue
        # especial cases
        if splited[0] in []:
            continue

        # new residue
        if splited[0] == "RESI":
            resname = str(splited[1])
            residues[resname] = {}

        # types weight
        if splited[0] == "MASS":
            typesMass[str(splited[2])] = float(splited[3])

        # defining a residue
        if resname is None:
            continue
        elif splited[0] == "ATOM":
            residues[resname][str(splited[1])] = {"type":str(splited[2]), "charge":float(splited[3])}


    fd = open(output, 'w')
    fd.write("# pdbparser simulation residues topology file\n")
    fd.write("# all atoms type and residues name are case sensitive\n\n")

    # lennard-Jones export
    fd.write("# atoms types and mass\n")
    fd.write("# mass: g/mole \n")
    fd.write("ATOMS_TYPE_MASS = {}\n")
    for key, item in typesMass.items():
        fd.write("ATOMS_TYPE_MASS['%s'] = %s\n"%(key, item))
    # Residues parameters
    for resname, values in residues.items():
        fd.write("\n# Residue %s \n"%resname)
        fd.write("%s = {}\n"%resname)
        for key, item in values.items():
            fd.write("%s['%s'] = %s\n"%(resname, key, item))



def convert_namd_param_file(param, output):
    """
    """
    # open file and get lines
    fd = open(param,'r')
    lines = fd.readlines()
    fd.close()

    # initialize flags
    LJFlag = False
    bondFlag = False
    angleFlag = False
    dihedralFlag = False
    improperFlag = False

    # initialize parameters
    lj = {}
    bond = {}
    angle = {}
    dihedral = {}
    improper = {}


    for line in lines:
        splited = line.strip().split()
        # empty line
        if not splited:
            continue
        # header
        if splited[0][0] == "*":
            continue
        # comment line
        if splited[0][0] == "!":
            continue
        # especial cases
        if splited[0] == "cutnb":
            continue

        # look for the start of params
        if splited[0] == "NONBONDED":
            LJFlag = True
            bondFlag = False
            angleFlag = False
            dihedralFlag = False
            improperFlag = False
            continue
        elif splited[0] == "BONDS":
            LJFlag = False
            bondFlag = True
            angleFlag = False
            dihedralFlag = False
            improperFlag = False
            continue
        elif splited[0] == "ANGLES":
            LJFlag = False
            bondFlag = False
            angleFlag = True
            dihedralFlag = False
            improperFlag = False
            continue
        elif splited[0] == "DIHEDRALS":
            LJFlag = False
            bondFlag = False
            angleFlag = False
            dihedralFlag = True
            improperFlag = False
            continue
        elif splited[0] == "IMPROPER":
            LJFlag = False
            bondFlag = False
            angleFlag = False
            dihedralFlag = False
            improperFlag = True
            continue
        elif splited[0] in ["CMAP", "HBOND"]:
            LJFlag = False
            bondFlag = False
            angleFlag = False
            dihedralFlag = False
            improperFlag = False
            continue

        # lennard-Jones parameters
        if LJFlag:
            try:
                lj[splited[0]] = {"eps":float(splited[2])*4.184, "rmin/2":float(splited[3])}
            except:
                print(splited)
                exit()

        # bond parameters
        if bondFlag:
            try:
                # preserve alphabetic order
                oneTwo = sorted([splited[0], splited[1]])
                key = "%s %s"%(oneTwo[0], oneTwo[1])
                bond[key] = {"kb":float(splited[2])*4.184, "b0":float(splited[3])}
            except:
                print(splited)
                exit()

        # angle parameters
        if angleFlag:
            try:
                # preserve alphabetic order
                oneThree = sorted([splited[0], splited[2]])
                key = "%s %s %s"%(oneThree[0], splited[1], oneThree[1])
                angle[key] = {"ktheta":float(splited[3])*4.184, "theta0":float(splited[4])}
            except:
                print(splited)
                exit()

        # dihedral parameters
        if dihedralFlag:
            try:
                # preserve alphabetic order
                oneFour = sorted([splited[0], splited[3]])
                key = "%s %s %s %s"%(oneFour[0], splited[1], splited[2], oneFour[1])
                # same dihedral term can have different multiplicity
                if key in dihedral:
                    dihedral[key][float(splited[5])] = {"kchi":float(splited[4])*4.184,"n":float(splited[5]), "delta":float(splited[6]) }
                else:
                    dihedral[key] = {float(splited[5]):{"kchi":float(splited[4])*4.184,"n":float(splited[5]), "delta":float(splited[6]) }}
            except:
                print(splited)
                exit()

        # improper parameters
        if dihedralFlag:
            try:
                key = "%s %s %s %s"%(splited[0], splited[1], splited[2], splited[3])
                improper[key] = {"kpsi":float(splited[4])* 4.184,"psi0":float(splited[5])}
            except:
                print(splited)
                exit()

    fd = open(output, 'w')
    fd.write("# pdbparser simulation parameters for NAMD forcefield terms\n")
    fd.write("# all atoms type are case sensitive\n\n")

    # lennard-Jones export
    fd.write("# Lennard-Jones parameters\n")
    fd.write("# V(Lennard-Jones) = Eps,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]\n")
    fd.write("# epsilon: kj/mole --> Eps,i,j = sqrt(eps,i * eps,j)\n")
    fd.write("# Rmin/2: A, Rmin,i,j = Rmin/2,i + Rmin/2,j\n")
    fd.write("LJ = {}\n")
    for key, item in lj.items():
        fd.write("LJ['%s'] = %s\n"%(key, item))
    # bond parameters
    fd.write("\n# bonds parameters\n")
    fd.write("# V(bond) = Kb(b - b0)**2\n")
    fd.write("# Kb: kj/mole/A**2\n")
    fd.write("# b0: A\n")
    fd.write("BOND = {}\n")
    for key, item in bond.items():
        fd.write("BOND['%s'] = %s\n"%(key, item))
    # angle parameters
    fd.write("\n# angles parameters\n")
    fd.write("# V(angle) = Ktheta(Theta - Theta0)**2\n")
    fd.write("# Ktheta: kj/mole/rad**2\n")
    fd.write("# Theta0: degrees\n")
    fd.write("ANGLE = {}\n")
    for key, item in angle.items():
        fd.write("ANGLE['%s'] = %s\n"%(key, item))
    # dihedral parameters
    fd.write("\n# dihedral parameters\n")
    fd.write("# V(dihedral) = Kchi(1 + cos(n(chi) - delta))\n")
    fd.write("# Kchi: kj/mole\n")
    fd.write("# n: multiplicity\n")
    fd.write("# delta: degrees\n")
    fd.write("DIHEDRAL = {}\n")
    for key, item in dihedral.items():
        fd.write("DIHEDRAL['%s'] = %s\n"%(key, item))
    # improper parameters
    fd.write("\n# improper parameters\n")
    fd.write("# V(improper) = Kpsi(psi - psi0)**2\n")
    fd.write("# Kpsi: kj/mole/rad**2\n")
    fd.write("# psi0: degrees\n")
    fd.write("IMPROPER = {}\n")
    for key, item in improper.items():
        fd.write("IMPROPER['%s'] = %s\n"%(key, item))
    # close file
    fd.close()



class Thermostat(object):
    """
    Mother class of all thermostats
    """
    def __init__(self, simulation, *args, **kwargs):

        self.__simulation__ = simulation
        assert isinstance(self.__simulation__, Simulation)

        # get simulation parameters
        self.__temperature__ = self.__simulation__.temperature
        self.__timeStep__ = float(self.__simulation__.timeStep)

    @property
    def simulation(self):
        return self.__simulation__

    @property
    def timeStep(self):
        return self.__timeStep__

    @property
    def temperature(self):
        return self.__temperature__

    def set_temperature(self, temperature):
        self.__temperature__ = temperature


    def get_temperature(self):
        return self.__temperature__


    def set_time_step(self, timestep):
        self.__timeStep__ = timestep


    def get_time_step(self):
        return self.__timeStep__


    def set_simulation(self, simulation):
        self.__simulation__ = simulation


    def get_simulation(self):
        return self.__simulation__


    def limit_velocity(self):
        # limit all velocities > limit A/fs to limit
        if self.get_simulation().limitVelocities is not None:
            velSquared = np.add.reduce(self.get_simulation().velocity**2,1)
            limSquared = self.get_simulation().limitVelocities**2
            toLimitIdx = np.where(velSquared > limSquared)[0]
            if not len(toLimitIdx):
                return
            self.get_simulation().velocity[toLimitIdx] *= np.array(limSquared/velSquared[toLimitIdx]).reshape(-1,1)

    def rescale_velocity(self):
        pass


class IsokineticThermostat(Thermostat):
    """
    Such a thermostat cannot be used to conduct a simulation in the canonical ensemble,
    but is perfectly fine to use in a warmup or initialization phase.

    Usage: IsokineticThermostat(simulation taut, fixcm)

    simulation
        append a simulation instance
    """
    def __init__(self, simulation, *args, **kwargs):
        # The base class constructor.
        super(IsokineticThermostat,self).__init__(simulation, *args, **kwargs)


    def rescale_velocity(self):
        """ simple velocity rescaling """
        self.limit_velocity()
        temp = self.get_simulation().boltzmannTemperature
        rescalingFactor =  np.sqrt(self.get_temperature()/temp)

        # rescale velocities
        self.get_simulation().velocity *= rescalingFactor


class BerendsenThermostat(Thermostat):
    """
    """
    def __init__(self, simulation, rise_time = None, *args, **kwargs):
        # The base class constructor.
        super(BerendsenThermostat,self).__init__(simulation, *args, **kwargs)
        # time coupling
        if rise_time is None:
            self.__riseTime = 2.0*self.get_time_step()
        else:
            self.__riseTime = float(rise_time)
        assert self.__riseTime>=self.get_time_step(), "rise_time must be > simulation's time_step"


    def set_rise_time(self, rise_time):
        self.__riseTime = float(rise_time)
        assert self.__riseTime>=self.get_time_step(), "rise_time must be > simulation's time_step"


    def get_rise_time(self):
        return self.__riseTime


    def rescale_velocity(self):
        """ simple velocity rescaling """
        # rescale all velocities > 1A/fs to 1
        self.limit_velocity()

        temp = self.get_simulation().boltzmannTemperature
        rescalingFactor =  np.sqrt( 1 + (self.get_time_step()/self.get_rise_time()) * (self.get_temperature()/temp-1) )

        # limit the velocity scaling to reasonable values
        if rescalingFactor > 1.1:
            rescalingFactor = 1.1
        if rescalingFactor < 0.9:
            rescalingFactor = 0.9
        # rescale velocities
        self.get_simulation().velocity *= rescalingFactor

        vel = np.add.reduce(self.get_simulation().velocity**2,1)
        meanVel = np.sum(vel)/self.get_simulation().numberOfAtoms
        maxIdx = vel.argmax()
        minIdx = vel.argmin()
        maxVel = vel[maxIdx]
        minVel = vel[minIdx]

        #print("target temp:", self.get_temperature())
        #print("found temp: ", self.get_simulation().boltzmannTemperature)
        #print("mean velocities: ", meanVel)
        #print("min velocity: ", minVel, minIdx)
        #print("max velocity: ", maxVel, maxIdx)
        #print("rescaling Factor: ", rescalingFactor, '\n')
        #print(self.get_simulation().velocity)





class Simulation(object):
    """
    The mother class of all simulate classes
    """
    __defaults__ = {}
    __defaults__["pdb"]                        = pdbparser
    __defaults__["boxVectors"]                 = None
    __defaults__["temperature"]                = 300
    __defaults__["pressure"]                   = 1.01325 * 10**5 # in pa=kg/m/s2
    __defaults__["numberOfSteps"]              = 1000
    __defaults__["outputFrequency"]            = 10
    __defaults__["outputPath"]                 = os.path.join(os.path.expanduser("~"),"pdbparserSimulation.xyz")
    __defaults__["exportInitialConfiguration"] = False
    __defaults__["foldCoordinatesIntoBox"]     = False
    __defaults__["timeStep"]                   = 1 # in fs
    __defaults__["simulationThermostat"]       = BerendsenThermostat
    __defaults__["minimizationThermostat"]     = IsokineticThermostat
    __defaults__["limitVelocities"]            = 1.0 # in A/fs
    __defaults__["masses"]                     = None
    __defaults__["charges"]                    = None
    __defaults__["bonds"]                      = None
    __defaults__["angles"]                     = None
    __defaults__["dihedrals"]                  = None
    __defaults__["improper"]                   = None
    __defaults__["logStatus"]                  = True
    __defaults__["logExport"]                  = True


    def __init__(self, pdb, *args, **kwargs):
        # initialize self._pdb
        object.__setattr__(self, "_pdb", pdbparser())

        # set pdb
        self.set_pdb(pdb)

        # set kwargs attributes
        for kwarg, value in kwargs.items():
            self.__setattr__(kwarg, value)

        # get default attributes
        self.initialize_default_attributes()

        # forcefield parameters
        try:
            self.set_forcefield_parameters()
        except:
            Logger.warn("Default forcefield parameters doesn't contain all used atoms type.")


    def __setattr__(self, name, value):
        if name == "_pdb":
            Logger.error("attribute %r is protected, user set_pdb method instead"%name)
        else:
            object.__setattr__(self, name, value)


    def initialize_default_attributes(self):
        """
        """
        # set constants
        self.__coulombConstant__ = 10**-3 * __convert_kn_mol__*__coulombConstant__ # in Coulomb-2 * kiloNewton * m**2
        self.__boltzmannConstant__ =  __convert_j__*__boltzmannConstant__ # in g mol^-1 A^2 fs^-2 K-1

        # get parameters
        self.__LJ__ = LJ
        self.__BOND__ = BOND
        self.__ANGLE__ = ANGLE
        self.__DIHEDRAL__ = DIHEDRAL
        self.__IMPROPER__ = IMPROPER

        # get output arguments
        if not hasattr(self, "outputPath"):
            self.outputPath = self.__defaults__["outputPath"]
        try:
            with open(self.outputPath) : os.remove(self.outputPath)
        except IOError:
            pass

        # output frequency
        if not hasattr(self, "outputFrequency"):
            self.outputFrequency = self.__defaults__["outputFrequency"]
        assert isinstance(self.outputFrequency, int)
        assert self.outputFrequency > 0

        # numberOfSteps
        if not hasattr(self, "numberOfSteps"):
            self.numberOfSteps = self.__defaults__["numberOfSteps"]
        assert isinstance(self.numberOfSteps, int)
        assert self.numberOfSteps > 0

        # numberOfSteps
        if not hasattr(self, "timeStep"):
            self.timeStep = self.__defaults__["timeStep"]
        assert isinstance(self.timeStep, (float, int))
        assert self.timeStep > 0

        # logStatus
        if not hasattr(self, "logStatus"):
            self.logStatus = self.__defaults__["logStatus"]
        assert isinstance(self.logStatus, bool)

        # logStatus
        if not hasattr(self, "logExport"):
            self.logExport = self.__defaults__["logExport"]
        assert isinstance(self.logExport, bool)

        # export initial configuration
        if not hasattr(self, "exportInitialConfiguration"):
            self.exportInitialConfiguration = self.__defaults__["exportInitialConfiguration"]
        assert isinstance(self.exportInitialConfiguration, bool)

        # export initial configuration
        if not hasattr(self, "boxVectors"):
            self.boxVectors = self.__defaults__["boxVectors"]
        if self.boxVectors is None:
            self.simulationBox = InfiniteBoundaries()
        else:
            self.simulationBox = PeriodicBoundaries()
        self.simulationBox.set_vectors(self.boxVectors)

        # logStatus
        if not hasattr(self, "foldCoordinatesIntoBox"):
            self.foldCoordinatesIntoBox = self.__defaults__["foldCoordinatesIntoBox"]
        assert isinstance(self.foldCoordinatesIntoBox, bool)

        # temperature
        if not hasattr(self, "temperature"):
            self.temperature = self.__defaults__["temperature"]
        assert isinstance(self.temperature, (float, int))
        assert self.temperature > 0

        # simulation ensemble
        if not hasattr(self, "simulationThermostat"):
            self.simulationThermostat = self.__defaults__["simulationThermostat"](self)
        else:
            self.simulationThermostat = self.simulationThermostat()
            assert isinstance(self.simulationThermostat, Thermostat)
            self.simulationThermostat.set_simulation(self)

        # minimization ensemble
        if not hasattr(self, "minimizationThermostat"):
            self.minimizationThermostat = self.__defaults__["minimizationThermostat"](self)
        else:
            self.minimizationThermostat = self.minimizationThermostat()
            assert isinstance(self.minimizationThermostat, Thermostat)
            self.minimizationThermostat.set_simulation(self)
        # limit velocities
        if not hasattr(self, "limitVelocities"):
            self.limitVelocities = self.__defaults__["limitVelocities"]
        if self.limitVelocities is not None:
            assert isinstance(self.limitVelocities, (float, int))
            assert self.limitVelocities > 0
        # limit pressure
        if not hasattr(self, "pressure"):
            self.pressure = self.__defaults__["pressure"]
        assert isinstance(self.pressure, (float, int))
        assert self.pressure > 0
        self.pressure *= __convert_pa__

        # atoms
        self.atomsName = get_records_attribute_values(self._pdb.indexes, self._pdb, "atom_name")
        self.atomsResidue = get_records_attribute_values(self._pdb.indexes, self._pdb, "residue_name")
        self.elements = get_records_attribute_values(self._pdb.indexes, self._pdb, "element_symbol")
        self.atomsType = [el.lower() for el in self.elements] # just here we should use lower because default forcefield types are all in lower case
        self.weights = np.array(get_records_database_property_values(self._pdb.indexes, self._pdb, "atomicWeight"))
        self.atomsCharge = np.zeros(len(self._pdb))
        self.numberOfAtoms = len(self.elements)

        # get bonds, angles, dihedrals
        C = Connectivity(self._pdb)
        self.bonds = C.get_bonds()[1]
        self.angles = C.get_angles()
        self.dihedrals = C.get_dihedrals()
        self.nBondsThreshold = self.find_upto_4_bonds_away()

        # coordinates
        self.coordinates = self.simulationBox.fold_real_array(get_coordinates(self._pdb.indexes, self._pdb))
        self.coordinates_0 = copy.deepcopy(self.coordinates)
        self.center = np.sum(self.coordinates,0)/len(self._pdb)

        # velocity and momentum
        self.velocity = np.zeros((len(self.elements),3))

        # temperatures
        self.boltzmannTemperature = 0

        # pressure
        self.virialStress = np.zeros((3,3))
        self.internalPressure =  np.zeros((3,3))

        # center of mass
        self.centerOfMass = np.copy(self.coordinates)
        self.centerOfMass[:,0] *= self.weights
        self.centerOfMass[:,1] *= self.weights
        self.centerOfMass[:,2] *= self.weights
        self.centerOfMass = np.sum(self.centerOfMass,0)/np.sum(self.weights)

        # final step force and energy
        self.force = np.zeros((len(self.elements), 3))
        self.potentialEnergy = np.zeros((len(self.elements)))

        # intialize trajectory
        self.trajectory = np.empty((len(self.elements), 3, self.numberOfSteps))


    def initialize_velocities(self, temperature):
        """
        Generates a random velocities distribution.
        the distribution center is the origin, so no initial drift
        m*v**2 = 3*Kb*T
        at each direction m*vx**2 = k*T
        """
        sigma = np.sqrt((temperature*self.__boltzmannConstant__)/self.weights)
        self.velocity = np.array([ np.random.normal(loc = 0, scale = sig, size = (3)) for sig in sigma ])
        # update boltzamnn temperature
        self.update_boltzmann_temperature()

#         boltzmannEnergy = 3./2.*self.__boltzmannConstant__*self.temperature
#         meanEc = np.sum(0.5*np.add.reduce(self.velocity**2,1)*self.weights)/self.numberOfAtoms
#         print(boltzmannEnergy, meanEc)
#         exit()


    def set_forcefield_parameters(self):
        self.set_lennard_jones_parameters()
        self.set_bonds_parameters()
        self.set_angles_parameters()
        self.set_dihedrals_parameters()


    def set_lennard_jones_parameters(self):
        # lennard-jones parameters = []
        self.lennardJones_eps = np.array([self.__LJ__[el]["eps"] for el in self.atomsType])*__convert_kj_mol__
        self.lennardJones_rmin_2 = np.array([self.__LJ__[el]["rmin/2"] for el in self.atomsType])


    def set_bonds_parameters(self):
        # bonds parameters
        self.bonds_kb = []
        self.bonds_b0 = []
        self.bonds_indexes = []
        for idx in range(len(self.atomsType)):
            self.bonds_indexes.extend( [[idx, item] for item in self.bonds[idx]] )
            element = self.atomsType[idx]
            bonds = [ self.atomsType[idx] for idx in self.bonds[idx]]
            keys = [sorted([element,bonded]) for bonded in bonds]
            keys = ["%s %s"%(key[0], key[1]) for key in keys]
            for key in keys:
                self.bonds_kb.append( self.__BOND__[key]["kb"] )
                self.bonds_b0.append( self.__BOND__[key]["b0"] )
        self.bonds_kb = np.array(self.bonds_kb)*__convert_kj_mol__
        self.bonds_b0 = np.array(self.bonds_b0)
        self.bonds_indexes = np.array(self.bonds_indexes)


    def set_angles_parameters(self):
        # angles parameters
        self.angles_ktheta = []
        self.angles_theta0 = []
        self.angles_indexes = []
        for angleIndexes in self.angles:
            elements = [ self.atomsType[idx] for idx in angleIndexes]
            oneThree = sorted([elements[0],elements[2]])
            key = "%s %s %s"%(oneThree[0], elements[1], oneThree[1])
            self.angles_theta0.append( self.__ANGLE__[key]["theta0"] )
            self.angles_ktheta.append( self.__ANGLE__[key]["ktheta"] )
            self.angles_indexes.append(angleIndexes)
        self.angles_theta0 = np.array(self.angles_theta0)*np.pi/180.
        self.angles_ktheta = np.array(self.angles_ktheta)*__convert_kj_mol__
        self.angles_indexes = np.array(self.angles_indexes)


    def set_dihedrals_parameters(self):
        # angles parameters
        self.dihedrals_kchi = []
        self.dihedrals_n = []
        self.dihedrals_delta = []
        self.dihedrals_indexes = []
        for dihedralIndexes in self.dihedrals:
            elements = [ self.atomsType[idx] for idx in dihedralIndexes]
            oneFour = sorted([elements[0],elements[3]])
            key = "%s %s %s %s"%(oneFour[0], elements[1], elements[2], oneFour[1])
            # same set of 4 types can have different parameters for more complex potential
            for k, param in self.__DIHEDRAL__[key].items():
                self.dihedrals_indexes.append(dihedralIndexes)
                self.dihedrals_kchi.append(param["kchi"])
                self.dihedrals_n.append(param["n"])
                self.dihedrals_delta.append(param["delta"])

        self.dihedrals_indexes = np.array(self.dihedrals_indexes)
        self.dihedrals_Kchi = np.array(self.dihedrals_kchi)*__convert_kj_mol__
        self.dihedrals_n = np.array(self.dihedrals_n)
        self.dihedrals_delta = np.array(self.dihedrals_delta)*np.pi/180.


    def load_parameters_file(self, parametersFile):
        """
        """
        import imp
        param = imp.load_source('', parametersFile)
        # update forcefield parameters
        self.__LJ__ = param.LJ
        self.__BOND__ = param.BOND
        self.__ANGLE__ = param.ANGLE
        self.__DIHEDRAL__ = param.DIHEDRAL
        self.__IMPROPER__ = param.IMPROPER


    def load_topology_file(self, topologyFile):
        """
        """
        import imp
        top = imp.load_source('', topologyFile)
        residues = [v for v in dir(top) if not v.startswith('_')]
        # update atoms parameters
        self.atomsType = [getattr(top, self.atomsResidue[idx])[self.atomsName[idx]]["type"] for idx in range(len(self.atomsName))]
        self.atomsCharge = np.array([getattr(top, self.atomsResidue[idx])[self.atomsName[idx]]["charge"] for idx in range(len(self.atomsName))])


    def find_upto_4_bonds_away(self):
        """
        """
        # recreate all bonds to all atoms
        bonds = copy.deepcopy(self.bonds)
        for idx in range(len(bonds)):
            for bondsIdx in bonds[idx]:
                bonds[bondsIdx].append(idx)
        bonds = [list(set(item)) for item in bonds]

        # create nbondsAway list
        nbondsAway = []
        for idx in range(len(bonds)):
            firstBond = bonds[idx]
            bondedList = copy.deepcopy(firstBond)
            for secondBond in firstBond:
                bondedList.extend(bonds[secondBond])
                for thirdBond in bonds[secondBond]:
                    bondedList.extend(bonds[thirdBond])
                    for fourthBond in bonds[thirdBond]:
                        bondedList.extend(bonds[fourthBond])

            bondedList = list( set(bondedList)-set(range(idx+1)) )
            nbondsAway.append(bondedList)

        for idx in range(len(nbondsAway)):
            item = list(set(nbondsAway[idx]))
        nbondsAway = [list(set(item)) for item in nbondsAway]

#         for idx in range(len(nbondsAway)):
#             print(get_records_attribute_values(self._pdb.indexes, self._pdb, "atom_name"))
#             print(get_records_attribute_values([idx], self._pdb, "atom_name"))
#             print(get_records_attribute_values(nbondsAway[idx], self._pdb, "atom_name"))
#             print
#
#         exit()
        return nbondsAway


    def set_pdb(self, pdb):
        """
        set the constructed pdb.\n

        :Parameters:
            #. pdb (pdbparser): The pdb instance replacing the constructed self._pdb.
        """
        assert isinstance(pdb, pdbparser)
        object.__setattr__(self, "_pdb", pdb)


    def status(self, step, numberOfSteps, stepIncrement = 1, mode = True, logEvery = 10, methodName = "SIMULATION"):
        """
        This method is used to log construction status.\n

        :Parameters:
            #. step (int): The current step number
            #. numberOfSteps (int): The total number of steps
            #. stepIncrement (int): The incrementation between one step and another
            #. mode (bool): if False no status logging
            #. logEvery (float): the frequency of status logging. its a percent number.
        """

        if not mode:
            return
        if step-1 == -1:
            return
        actualPercent = int( float(step)/float(numberOfSteps)*100)
        previousPercent = int(float(step-stepIncrement)/float(numberOfSteps)*100)

        if actualPercent/logEvery != previousPercent/logEvery:
            #print(self.coordinates)
            Logger.info("%s --> total energy: %10.5f - %s%% completed. %s left out of %s" %(methodName, np.sum(self.potentialEnergy), actualPercent, numberOfSteps-step, numberOfSteps))


    def get_trajectory(self):
        """
        get the simulated pdb trajectory.\n

        :Returns:
            #. self.trajectory (numpy.array): The records positions through all simulation time.
        """
        return self.trajectory


    def update_boltzmann_temperature(self):
        """
        calculate system temperature from actual velocities
        """
        meanEc_2 = np.sum(np.add.reduce(self.velocity**2,1)*self.weights)/self.numberOfAtoms
        self.boltzmannTemperature = meanEc_2/(3*self.__boltzmannConstant__)


    def get_current_frame_pdb(self, fold = False):
        """
        """
        # get a pdb copy
        pdb = self._pdb.get_copy()

        if fold:
            coordinates = self.simulationBox.fold_real_array(self.coordinates)
        else:
            coordinates = self.coordinates

        set_coordinates(pdb.indexes, pdb, coordinates)
        return pdb


    def export_current_coordinates(self, step):
        """
        """
        if not step or step % self.outputFrequency:
            return

        if self.logExport:
            Logger.info("SIMULATION --> total energy: %s, writing step %r to file %r" %(np.sum(self.potentialEnergy), step, self.outputPath))

        # open file
        fd = open(self.outputPath, 'a')

        # write snapshot
        fd.write("%s\n"%len(self.atomsName))
        fd.write("pdbparser simulation. total time %s, snapshot of step %s\n"%(self.numberOfSteps,step) )
        if self.foldCoordinatesIntoBox:
            coordinates = self.simulationBox.fold_real_array(self.coordinates)
        else:
            coordinates = self.coordinates

        for idx in self._pdb.indexes:
            atName = str(self.atomsName[idx]).strip().rjust(5)
            coords = "%10.5f"%coordinates[idx][0] + " %10.5f"%coordinates[idx][1] + " %10.5f"%coordinates[idx][2]
            fd.write( atName + " " + coords + "\n")

        # close file
        fd.close()


    def visualize_trajectory(self, path, vmdAlias = None):
        """
        """
        if vmdAlias is None:
            from pdbparser.pdbparser import get_vmd_path
            vmdAlias = get_vmd_path()

        try:
            os.system( "%s %s" %(vmdAlias, path) )
        except:
            Logger.warn('vmd alias %r defined is not correct' %vmdAlias)
            raise


    def electrostatic_term(self, distances, particuleCharge, charges, omit):
        """
        electrostatic force
        F = 1/4piEps0 * q1*q2/r**2
        """
        # put to zero all up to 4 bonds away
        charges[omit] = 0

        # calculate potential
        potentialEnergy = self.__coulombConstant__* (particuleCharge*charges/distances)
        # calculate forces
        force = -potentialEnergy/distances

        return potentialEnergy, force


    def LennardJones_12_6_CHARMM(self, r, rmin, eps, omit):
        """
        Lennard-Jones potential power 12 - power 6 format
        V(Lennard-Jones) = Eps,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]
        epsilon: 10**-4*kj/mole, Eps,i,j = sqrt(eps,i * eps,j)
        Rmin/2: A, Rmin,i,j = Rmin/2,i + Rmin/2,j
        """
        # calculate Rmin/distances
        ratio = np.divide(rmin, r)

        # put to zero all up to 4 bonds away
        ratio[omit] = 0

        # calculate potential
        potentialEnergy = eps * (ratio**12 - 2*ratio**6)

        # calculate forces
        force = -12.0*eps/rmin * (ratio**13 - ratio**7)

        return potentialEnergy, force


    def bond_CHARMM(self, r, kb, b0):
        """
        bonds potential
        V(bond) = Kb(b - b0)**2
        b0: A
        """
        b = np.sqrt(np.add.reduce(r**2,1))

        # calculate b - b0
        b_b0 = b - b0

        # calculate potential
        potentialEnergy = kb*b_b0**2

        # normalize direction
        r /= b[:, np.newaxis]

        # calculate forces
        force = r * (2*kb*b_b0)[:, np.newaxis]

        return potentialEnergy, force


    def angle_CHARMM(self, rij, rkj, ktheta, theta0):
        """
        angles potential
        V(angle) = Ktheta(Theta - Theta0)**2
        Ktheta: kcal/mole/rad**2
        Theta0: rad
        """
        # normalize rij and rkj
        norm_of_rij = np.sqrt((rij ** 2).sum(1))
        norm_of_rkj = np.sqrt((rkj ** 2).sum(1))
        normalized_rij = rij/ norm_of_rij[:, np.newaxis]
        normalized_rkj = rkj/ norm_of_rkj[:, np.newaxis]

        # caluclate the normalized r_perpendiculare to rij and rkj
        rpp = np.cross(normalized_rkj,normalized_rij)

        # calculate theta-theta0
        rij_dot_rkj = np.sum(normalized_rij*normalized_rkj, 1)
        angles = np.arccos( rij_dot_rkj )
        Theta_Theta0 = angles - theta0

        # calculate potential energy
        potentialEnergy = ktheta*Theta_Theta0**2

        # calculate nabla_ij and nabla_jk
        nabla_ij = -1./norm_of_rij[:, np.newaxis] * np.cross(normalized_rij, rpp)
        nabla_kj = 1./norm_of_rkj[:, np.newaxis] * np.cross(normalized_rkj, rpp)

        # calculate forces
        thetaForce = 2*(ktheta*Theta_Theta0)[:, np.newaxis]
        forcei = thetaForce * nabla_ij
        forcej = thetaForce * (-nabla_ij-nabla_kj)
        forcek = thetaForce * nabla_kj

        return potentialEnergy, [forcei,forcej,forcek]


    def dihedral_CHARMM(self, rij, rkj, rlk, n, kchi, delta):
        """
        dihedral potential
        V(dihedral) = Kchi(1 + cos(n(chi) - delta))
        Kchi: kcal/mole
        n: multiplicity
        delta: rad
        """
        # normalize rij and rkj, rlk
        norm_of_rij = np.sqrt((rij ** 2).sum(1))
        norm_of_rkj = np.sqrt((rkj ** 2).sum(1))
        norm_of_rlk = np.sqrt((rlk ** 2).sum(1))
        normalized_rij = rij / norm_of_rij[:, np.newaxis]
        normalized_rkj = rkj / norm_of_rkj[:, np.newaxis]
        normalized_rlk = rlk / norm_of_rlk[:, np.newaxis]

        # calculate t and u
        t = np.cross(rij, rkj)
        u = np.cross(rlk, rkj)
        norm_of_t = np.sqrt((t ** 2).sum(1))
        norm_of_u = np.sqrt((u ** 2).sum(1))
        normalized_t = t/ norm_of_t[:, np.newaxis]
        normalized_u = u/ norm_of_u[:, np.newaxis]

        # calculate chi and nchi_delta
        chi = np.arccos( np.sum(normalized_t*normalized_u, 1) )
        nchi_delta = n*chi-delta

        # calculate potential energy
        potentialEnergy = kchi * ( 1+np.cos(nchi_delta) )

        # calculate nabla_t and nabla_u
        nabla_t = 1./norm_of_t[:, np.newaxis] * np.cross(normalized_t, normalized_rkj)
        nabla_u = -1./norm_of_u[:, np.newaxis] * np.cross(normalized_u, normalized_rkj)

        # calculate nabla_j
        nabla_i = np.array([ nabla_t[:,1]*(-rkj[:,2]) + nabla_t[:,2]*( rkj[:,1]),
                             nabla_t[:,0]*( rkj[:,2]) + nabla_t[:,2]*(-rkj[:,0]),
                             nabla_t[:,0]*(-rkj[:,1]) + nabla_t[:,1]*( rkj[:,0]) ])

        nabla_j = np.array([ nabla_t[:,1]*(rkj[:,2]-rij[:,2]) + nabla_t[:,2]*(rij[:,1]-rkj[:,1]) + nabla_u[:,1]*(-rlk[:,2]) + nabla_u[:,2]*( rlk[:,1]),
                             nabla_t[:,0]*(rij[:,2]-rkj[:,2]) + nabla_t[:,2]*(rkj[:,0]-rij[:,0]) + nabla_u[:,0]*( rlk[:,2]) + nabla_u[:,2]*(-rlk[:,0]),
                             nabla_t[:,0]*(rkj[:,1]-rij[:,1]) + nabla_t[:,1]*(rij[:,0]-rkj[:,0]) + nabla_u[:,0]*(-rlk[:,1]) + nabla_u[:,1]*( rlk[:,0])])

        nabla_l = np.array([ nabla_u[:,1]*(-rkj[:,2]) + nabla_u[:,2]*( rkj[:,1]),
                             nabla_u[:,0]*( rkj[:,2]) + nabla_u[:,2]*(-rkj[:,0]),
                             nabla_u[:,0]*(-rkj[:,1]) + nabla_u[:,1]*( rkj[:,0]) ])

        # calculate forces
        chiForce = -n*kchi*np.sin(nchi_delta)
        forcei = np.reshape( chiForce * nabla_i, (-1,3))
        forcej = np.reshape( chiForce * nabla_j, (-1,3))
        forcek = np.reshape( chiForce * (-nabla_i-nabla_j-nabla_l), (-1,3))
        forcel = np.reshape( chiForce * nabla_l, (-1,3))

        return potentialEnergy,  [forcei,forcej,forcek,forcel]


    def update_virial_stress(self, rijVector, fijVector):
        """ calculaes the stress tensor
            stress =[ xx xy xz
                      yx yy yz
                      zx zy zz ]
        """
        # calculate the dot product and add it
        stress = np.dot(fijVector.T, rijVector)

        self.virialStress += stress


    def calculate_pressure(self):
        """ calculates the pressure after each step """
        #KE = np.sum(np.add.reduce(self.velocity**2,1)*self.weights)/self.numberOfAtoms
        KE =  np.dot(self.velocity.T, self.velocity*self.weights.reshape(-1,1) )
        self.internalPressure = (KE - self.virialStress) / self.simulationBox.get_box_volume()


    def simple_integration(self):
        """
        """
        # calculate acceleration
        acceleration = np.divide(self.force , self.weights.reshape(-1,1) )

        # update velocity
        self.velocity += acceleration*self.timeStep

        # update coordinates
        self.coordinates += self.velocity*self.timeStep


    def verlet_integration(self):
        """
        """
        # calculate acceleration
        acceleration = np.divide(self.force , self.weights.reshape(-1,1) )

        # update velocity
        self.velocity += acceleration*self.timeStep

        # calculate new coordinates
        #newCoords = 2*self.coordinates - self.coordinates_0 + self.velocity*self.timeStep
        self.coordinates += self.velocity*self.timeStep

        # update position at time = t and t-1
        #self.coordinates_0 = copy.deepcopy(self.coordinates)
        #self.coordinates = newCoords


    def update_simulation_state(self):
        """
        """
        # http://www.ccp4.ac.uk/maxinf/mm4mx/PDFs/MM4MX.MD.Stote.pdf
        # self.weights is in uma, force is in Kj/mol
        # accleration calculation in A/fs**2 is straight forward with kcal to j conversion
        # acceleration unit conversion to A/fs2
        # force -- > kj/mole = 10**3 kg.m2/s2/mol --> 10**-4 g.A2/fs2/mol
        # rescale velocities for time = t-1
        self.simulationThermostat.rescale_velocity()
        # calculate new positions
        #self.verlet_integration()
        self.simple_integration()
        # update current state temperature
        self.update_boltzmann_temperature()


    def update_minimization_state(self):
        """
        """
        # http://www.ccp4.ac.uk/maxinf/mm4mx/PDFs/MM4MX.MD.Stote.pdf
        # self.weights is in uma, force is in Kj/mol
        # accleration calculation in A/fs**2 is straight forward with kcal to j conversion
        # acceleration unit conversion to A/fs2
        # force -- > kj/mole = 10**3 kg.m2/s2/mol --> 10**-4 g.A2/fs2/mol

        # calculate acceleration
        acceleration = np.divide(self.force , self.weights.reshape(-1,1) )

        # update velocity
        self.velocity = acceleration*self.timeStep
        # update current state temperature
        #self.update_boltzmann_temperature()

        # rescale velocities for time = t-1
        #self.minimizationThermostat.rescale_velocity()

        # calculate new positions
        self.coordinates += self.velocity*self.timeStep


    def simulate_step(self):
        """
        """
        # non bonded potentiel
        for idx in self._pdb.indexes[:-1]:
            #difference = self.coordinates[idx+1:,:] - self.coordinates[idx,:]
            difference = self.simulationBox.real_difference(realArray = self.coordinates[idx+1:,:], realVector = self.coordinates[idx,:])
            distances = np.sqrt(np.add.reduce(difference**2,1))

            # normalize difference
            difference /= distances[:, np.newaxis]

            # omit less that 4 bond distances
            omit = list(np.array(self.nBondsThreshold[idx])-idx-1)

            # calculate LennardJones term
            rmin = self.lennardJones_rmin_2[idx+1:] + self.lennardJones_rmin_2[idx]
            eps = np.sqrt(self.lennardJones_eps[idx+1:] * self.lennardJones_eps[idx])
            potentialEnergy, force = self.LennardJones_12_6_CHARMM(distances, rmin, eps, omit = omit)

            # update self.force and self.potentialEnergy
            self.potentialEnergy[idx+1:] += potentialEnergy
            self.potentialEnergy[idx] += np.sum(potentialEnergy)
            # convert difference to force
            forcevector = difference*force[:, np.newaxis]
            #self.force[idx+1:,:] -= forcevector
            #self.force[idx,:] += np.sum(forcevector,0)

            # calculate the electrostatic force
            potentialEnergy, force = self.electrostatic_term(distances, particuleCharge = self.atomsCharge[idx], charges = np.array(self.atomsCharge[idx+1:]), omit = omit)
            # update self.force and self.potentialEnergy
            self.potentialEnergy[idx+1:] += potentialEnergy
            self.potentialEnergy[idx] += np.sum(potentialEnergy)
            # convert difference to force
            forcevector += difference* force[:, np.newaxis]
            self.force[idx+1:,:] -= forcevector
            self.force[idx,:] += np.sum(forcevector,0)

            # calculate pressure virial term
            self.update_virial_stress(difference, forcevector)


        # calculate bonds
        if len(self.bonds_indexes):
            #r = self.coordinates[self.bonds_indexes[:,1],:] - self.coordinates[self.bonds_indexes[:,0],:]
            r = self.simulationBox.real_difference( realArray = self.coordinates[self.bonds_indexes[:,1],:],
                                                   realVector = self.coordinates[self.bonds_indexes[:,0],:] )
            potentialEnergy, force = self.bond_CHARMM(r, self.bonds_kb, self.bonds_b0 )
            # update self.force and self.potentialEnergy
            self.potentialEnergy[self.bonds_indexes[:,0]] += potentialEnergy
            self.potentialEnergy[self.bonds_indexes[:,1]] += potentialEnergy
            self.force[self.bonds_indexes[:,0],:] += force
            self.force[self.bonds_indexes[:,1],:] -= force

        # calculate angles
        if len(self.angles_indexes):
            #rij = self.coordinates[self.angles_indexes[:,0],:] - self.coordinates[self.angles_indexes[:,1],:]
            #rkj = self.coordinates[self.angles_indexes[:,2],:] - self.coordinates[self.angles_indexes[:,1],:]
            rij = self.simulationBox.real_difference( realArray = self.coordinates[self.angles_indexes[:,0],:],
                                                     realVector = self.coordinates[self.angles_indexes[:,1],:] )
            rkj = self.simulationBox.real_difference( realArray = self.coordinates[self.angles_indexes[:,2],:],
                                                     realVector = self.coordinates[self.angles_indexes[:,1],:] )
            potentialEnergy, forces = self.angle_CHARMM(rij, rkj, self.angles_ktheta, self.angles_theta0 )
            # update self.force and self.potentialEnergy
            self.potentialEnergy[self.angles_indexes[:,0]] += potentialEnergy
            self.potentialEnergy[self.angles_indexes[:,1]] += potentialEnergy
            self.potentialEnergy[self.angles_indexes[:,2]] += potentialEnergy
            self.force[self.angles_indexes[:,0],:] -= forces[0]
            self.force[self.angles_indexes[:,1],:] -= forces[1]
            self.force[self.angles_indexes[:,2],:] -= forces[2]

        # calculate dihedrals
        if len(self.dihedrals_indexes):
            #rij = self.coordinates[self.dihedrals_indexes[:,0],:] - self.coordinates[self.dihedrals_indexes[:,1],:]
            #rkj = self.coordinates[self.dihedrals_indexes[:,2],:] - self.coordinates[self.dihedrals_indexes[:,1],:]
            #rlk = self.coordinates[self.dihedrals_indexes[:,3],:] - self.coordinates[self.dihedrals_indexes[:,2],:]
            rij = self.simulationBox.real_difference( realArray = self.coordinates[self.dihedrals_indexes[:,0],:],
                                                     realVector = self.coordinates[self.dihedrals_indexes[:,1],:] )
            rkj = self.simulationBox.real_difference( realArray = self.coordinates[self.dihedrals_indexes[:,2],:],
                                                     realVector = self.coordinates[self.dihedrals_indexes[:,1],:] )
            rlk = self.simulationBox.real_difference( realArray = self.coordinates[self.dihedrals_indexes[:,3],:],
                                                     realVector = self.coordinates[self.dihedrals_indexes[:,2],:] )
            potentialEnergy, forces = self.dihedral_CHARMM(rij, rkj, rlk ,self.dihedrals_n, self.dihedrals_kchi, self.dihedrals_delta)
             # update self.force and self.potentialEnergy
            self.potentialEnergy[self.dihedrals_indexes[:,0]] += potentialEnergy
            self.potentialEnergy[self.dihedrals_indexes[:,1]] += potentialEnergy
            self.potentialEnergy[self.dihedrals_indexes[:,2]] += potentialEnergy
            self.potentialEnergy[self.dihedrals_indexes[:,3]] += potentialEnergy
            self.force[self.dihedrals_indexes[:,0],:] += forces[0]
            self.force[self.dihedrals_indexes[:,1],:] += forces[1]
            self.force[self.dihedrals_indexes[:,2],:] += forces[2]
            self.force[self.dihedrals_indexes[:,3],:] += forces[3]


    def minimize_steepest_descent(self, numberOfSteps = None):
        """
        """
        if numberOfSteps is None:
            numberOfSteps = self.numberOfSteps

        # write first configuration
        if self.exportInitialConfiguration:
            self.export_current_coordinates(1)
            firstStep = 2
        else:
            firstStep = 1

        # run minimization
        for step in xrange(firstStep,numberOfSteps+firstStep):
            # log status
            self.status(step, numberOfSteps, 1, mode = self.logStatus, logEvery = 10, methodName = "MINIMIZATION" )
            # reinitialize force and potential energy
            self.force = np.zeros((len(self.elements), 3))
            self.potentialEnergy = np.zeros((len(self.elements)))
            # minimize for step
            self.simulate_step()
            # update state
            self.update_minimization_state()
            # write snapshot
            self.export_current_coordinates(step)


    def simulate(self, numberOfSteps = None, initializeVelocities = True):
        """
        """
        # intialize velocities
        if initializeVelocities:
            Logger.info("SIMULATION --> Initializing velocities at temperature %s"%self.temperature)
            self.initialize_velocities(self.temperature)

        ## move pdb center to origin
        #self.coordinates -= self.centerOfMass
        if numberOfSteps is None:
            numberOfSteps = self.numberOfSteps

        # write first configuration
        if self.exportInitialConfiguration:
            self.export_current_coordinates(1)
            firstStep = 2
        else:
            firstStep = 1

        # run minimization
        for step in xrange(firstStep,numberOfSteps+firstStep):
            # log status
            self.status(step, numberOfSteps, 1, mode = self.logStatus, logEvery = 10, methodName = "SIMULATION" )
            # reinitialize force and potential energy
            self.force = np.zeros((len(self.elements), 3))
            self.potentialEnergy = np.zeros((len(self.elements)))

            # reinitialize pressure constants
            self.virialStress = np.zeros((3,3))

            # minimize for step
            self.simulate_step()

            # update state
            self.update_simulation_state()

            # calculate pressure
            self.calculate_pressure()
            #print("virialStress:\n ", self.virialStress)
            #print("pressure:\n ",self.internalPressure)
            #print(self.pressure, self.internalPressure[0][0],  self.internalPressure[1][1],  self.internalPressure[2][2])

            # write snapshot
            self.export_current_coordinates(step)
