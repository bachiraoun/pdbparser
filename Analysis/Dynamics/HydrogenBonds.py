"""
This module provides all Hydrogen bonds analysis classes.

.. inheritance-diagram:: pdbparser.Analysis.Dynamics.HydrogenBonds
    :parts: 2
"""
# standard libraries imports
from __future__ import print_function
from collections import Counter

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser.log import Logger
from pdbparser.Analysis.Core import Analysis, AxisDefinition
from pdbparser.Utilities.Information import get_records_database_property_values
from pdbparser.Utilities.Collection import correlation, get_data_weighted_sum
from pdbparser.Utilities.Database import get_element_property, is_element_property


class HydrogenBonds(Analysis):
    """
    Two electronegative atoms can be closer than their VdW radii would normally allow if one of
    them (the Donor) is bonded to a hydrogen, which is partially shared by the other (the Acceptor).

    Hydrogen bonds occur when a "donor" atom donates its covalently bonded hydrogen atom to
    an electronegative "acceptor" atom. The oxygen in -OH (e.g. Ser, Thr, Tyr), HOH, and the
    nitrogen in -NH3+ (as in Lys, Arg) or -NH- (as in the main chain peptide bond, Trp, His,
    Arg, nucleotide bases) are typical donors.
    The lone electron pairs on these same donors can serve as H bond acceptor sites. So can
    those on carbonyl oxygens =O (as in the main chain) or nitrogens with three covalent
    bonds =N- (as in His, Trp, or nucleotide bases). Lacking hydrogens, these latter cannot serve as donors.

    Jeffrey[2] categorizes H bonds with donor-acceptor distances of 2.2-2.5 A as "strong,
    mostly covalent", 2.5-3.2 A as "moderate, mostly electrostatic", 3.2-4.0 A as "weak,
    electrostatic" (page 12). Energies are given as 40-14, 15-4, and <4 kcal/mol respectively.
    Most H bonds in proteins are in the moderate category, strong H bonds requiring moieties or
    conditions that are rare within proteins. The hydrogen atoms in moderate H bonds often do
    not lie on the straight line connecting the donor to acceptor, so donor-acceptor distance
    slightly underestimates the length of the H bond (Jeffrey, p. 14). The mean donor-acceptor
    distances in protein secondary structure elements are close to 3.0 A, as are those between
    bases in Watson-Crick pairing (Jeffrey, pp. 191, 200).

    :Several parameters can be used to characterize the hydrogen bond:
        #. distance(Donor-Acceptor)
        #. distance(Hydrogen-Acceptor)
        #.  angle(Donor-Hydrogen-Acceptor)
        #.  angle(Hydrogen-Acceptor-Acceptor First bonded atom)
        #.  torsion(-Acceptor-Acceptor First bonded atom-)

    In this analysis we track the distance between Hydrogen and Acceptor atoms and the angle
    formed between the two vectors (donor,hydrogen) and (hydrogen,acceptor) to characterize a Hydrogen Bond.
    It is also possible to modelize a Hydrogen bond distance as core shell where a bond can survive infinitely
    encapsulated in other shells with finite survival time.

    :References Cited:
        #.  Martz, Eric; Help, Index & Glossary for Protein Explorer, http://www.umass.edu/microbio/chime/pe_beta/pe/protexpl/igloss.htm
        #. Jeffrey, George A.; An introduction to hydrogen bonding, Oxford University Press, 1997.

    :Parameters:
        #. trajectory (pdbTrajectory): pdbTrajectory instance.
        #. configurationsIndexes (list, set, tuple): List of selected indexes of configuration used to perform the analysis.
        #. acceptorAtomsIndexes (list, set, tuple): Selected acceptors atoms indexes. if none oxygen,
           nitrogen or fluorine element is selected awarning will be logged.
        #. hydrogenAtomsIndexes (list, set, tuple): Selected hydrogen atoms indexes. if none hydrogen
           element selected a warning will be logged
        #. donorsAtomsIndexes (list, set, tuple): Selected donors atoms indexes. must have the same
           length as hydrogenAtomsIndexes. Hydrogens and donors are coupled index wise.
        #. bondLength (float): The hydrogen bond core shell length in Angstrom. Must be bigger than 0.5 A. e.g. 2.7A
        #. thresholdTime (float): The threshold survival time in ps for a hydrogen bond needed before it is considered valid.
           If not reached before the bond get cut, the bond is discarded.
        #. bondAngleLimits (list, set, tuple): The Donor--Hydrogen--Acceptor angle lower and upper limits in degrees
           outside of which the Hydrogen bond is considered broken. e.g. (130,180)
        #. belowAngleToleranceTime (list, set, tuple): The time allowed for a bond angle to stay below the limit in pico-second. e.g. 0.05 ps
        #. bin (float): The bonds lengths histogram bin in Angstrom. Must be 0 < bin <= 1. e.g. 0.05A
        #. grouping (string, None): Must be a pdbparser record valid key. Hydrogen bonding of atoms in
           the same group are omitted. e.g. 'sequence_number' to omit hydrogen bond between atoms of the same residue.
           None is allowed for considering all bonds.
        #. toleranceShells (list, set, tuple): The hydrogen bond tolerance shells beyond the defined bondLength in Angstrom.
           Must be a list of positive numbers. e.g. [1.3, 0.5] the shells thickness add up to the defined bond
           distance. in this example the first shell is from 2.7A to 4A and the second shell from 4A to 4.5A
        #. toleranceTime (list, set, tuple): The time allowed for a bond distance to stay in a tolerance shell in pico-second.
           It must have the same length as toleranceShells. Every number in tolerance time list corresponds to a shell
           in toleranceShells list. e.g. [0.05, 0.01] a bond can stay 0.05ps in the first shell of 1.3A and 0.01 ps in the second shell of 0.5A.
        #. bondStartAtCore (boolean): Whether a bond can be found for the first time within the defined bondLength
           distance or in any other defined shell. e.g. True. In this case, only bonds that start within the core distance
           defined as bondLength are recorded and those that start in outer shells are ignored.
        #. bondStartWithinAngle (boolean): Whether a bond can be found for the first time between the allowed angle
           defined by the argument bondAngleLimits. e.g. True. In this case, only bonds that start within the allowed
           bond angle are recorded and those that start with an angle outside of the defined limits are ignored.
        #. smoothHfJumps (boolean): This option is used to smooth high frequency jumps. In other term it removes abrupt single
           back and forth jumps between shells and/or different acceptors. Bond length at jump time is corrected by the mean
           length value before and after the jump. e.g.True. High frequency jumps cut survival time of a bond in a shell.
    """

    def __init__(self, trajectory, configurationsIndexes,
                       donorsAtomsIndexes, hydrogenAtomsIndexes, acceptorAtomsIndexes,
                       bondLength, thresholdTime,
                       bondAngleLimits, belowAngleToleranceTime,
                       bin, grouping,
                       toleranceShells, toleranceTime,
                       bondStartAtCore, bondStartWithinAngle,
                       smoothHfJumps,
                       *args, **kwargs):
        # set trajectory
        super(HydrogenBonds,self).__init__(trajectory, *args, **kwargs)
        # set atoms selection
        self.donorsAtomsIndexes = self.get_atoms_indexes(donorsAtomsIndexes, sort=False, removeRedundancy=False)
        self.hydrogenAtomsIndexes = self.get_atoms_indexes(hydrogenAtomsIndexes, sort=False, removeRedundancy=False)
        self.acceptorAtomsIndexes = self.get_atoms_indexes(acceptorAtomsIndexes)
        assert len(self.donorsAtomsIndexes) == len(self.hydrogenAtomsIndexes), Logger.error("donorsAtomsIndexes and hydrogenAtomsIndexes must have the same size")
        assert len(self.hydrogenAtomsIndexes) == len(set(self.hydrogenAtomsIndexes)), Logger.error("hydrogenAtomsIndexes redundant indexes found")
        assert len(self.donorsAtomsIndexes), Logger.error("donorsAtomsIndexes can't be empty")
        assert len(self.hydrogenAtomsIndexes), Logger.error("hydrogenAtomsIndexes can't be empty")
        assert len(self.acceptorAtomsIndexes), Logger.error("acceptorAtomsIndexes can't be empty")
        assert len(self.donorsAtomsIndexes)==len(self.hydrogenAtomsIndexes), Logger.error("donorsAtomsIndexes and hydrogenAtomsIndexes must have the same length")
        # set steps indexes
        self.configurationsIndexes = self.get_trajectory_indexes(configurationsIndexes)
        self.numberOfSteps = len(self.configurationsIndexes)
        # initialize variables
        self.__initialize_variables__(grouping, bondLength,
                                      thresholdTime, bin,
                                      toleranceShells, toleranceTime,
                                      bondAngleLimits, belowAngleToleranceTime,
                                      bondStartAtCore, bondStartWithinAngle,
                                      smoothHfJumps)
        # initialize results
        self.__initialize_results__()


    def __initialize_variables__(self, grouping, bondLength,
                                 thresholdTime, bin,
                                 toleranceShells, toleranceTime,
                                 bondAngleLimits, belowAngleToleranceTime,
                                 bondStartAtCore, bondStartWithinAngle,
                                 smoothHfJumps):
        # set grouping
        if grouping is not None:
            assert isinstance(grouping, str), Logger.error("grouping must be a string convertible")
            grouping = str(grouping).lower()
            assert grouping in self.structure[0].keys(), Logger.error("grouping must be a pdbparser record key")
        self.grouping = grouping
        if self.grouping is None:
            self.groups = self.structure.indexes
        else:
            self.groups = [rec[self.grouping] for rec in self.structure.records]
        self.hydrogenAtomsGroup = [self.groups[idx] for idx in self.hydrogenAtomsIndexes]
        self.acceptorAtomsGroup = [self.groups[idx] for idx in self.acceptorAtomsIndexes]
        # set bondLength
        try:
            bondLength=float(bondLength)
        except:
            raise Logger.error("bondLength must be positive number")
        assert bondLength>0.5, Logger.error("bondLength must be bigger than 0.5")
        self.bondLength=bondLength
        # set thresholdTime
        try:
            thresholdTime=float(thresholdTime)
        except:
            raise Logger.error("thresholdTime must be a number")
        assert bondLength>=0, Logger.error("thresholdTime must be a positive number")
        self.thresholdTime=thresholdTime
        # set bondAngleLimits
        assert isinstance(bondAngleLimits, (list,set,tuple)), Logger.error("bondAngleLimits must be a lists")
        assert len(bondAngleLimits) ==2, Logger.error("bondAngleLimits must be a list of 2 items")
        try:
            bondAngleLimits= sorted([float(bondAngleLimits[0]),float(bondAngleLimits[1])])
        except:
            raise Logger.error("bondAngleLimits must be a list of positive 2 floats")
        assert bondAngleLimits[0]>=0, Logger.error("bondAngleLimits items must be a positive number")
        assert bondAngleLimits[1]<=180, Logger.error("bondAngleLimits items must be smaller than 180")
        self.bondAngleLimits=bondAngleLimits
        # set belowAngleToleranceTime
        try:
            belowAngleToleranceTime=float(belowAngleToleranceTime)
        except:
            raise Logger.error("belowAngleToleranceTime must be a number")
        assert belowAngleToleranceTime>0, Logger.error("bondAngleLimits must be bigger than 0")
        self.belowAngleToleranceTime=belowAngleToleranceTime
        # set tolerance shells and time
        if toleranceShells is None:
            toleranceShells = []
            toleranceTime   = []
        else:
            assert isinstance(toleranceShells, (list,set,tuple)), Logger.error("toleranceShells must be a list of positive floats")
            assert isinstance(toleranceTime, (list,set,tuple)), Logger.error("toleranceTime must be a list of positive floats")
            toleranceShells = list(toleranceShells)
            toleranceTime = list(toleranceTime)
        assert len(toleranceShells)==len(toleranceTime), Logger.error("toleranceShells and toleranceTime list must have the same number of items")
        try:
            toleranceShells = [float(item) for item in toleranceShells]
        except:
            raise Logger.error("toleranceShells must be a list of float positive numbers")
        try:
            toleranceTime = [float(item) for item in toleranceTime]
        except:
            raise Logger.error("toleranceTime must be a list of numbers")
        assert len(toleranceShells)==sum([1 for item in toleranceShells if item>=0]), Logger.error("toleranceShells must be a list of float positive numbers")
        assert len(toleranceTime)==sum([1 for item in toleranceTime if item>=0]), Logger.error("toleranceTime must be a list of float positive numbers")
        self.toleranceShells = []
        self.toleranceTime = []
        for idx in range(len(toleranceShells)):
            if toleranceShells[idx]>0 and toleranceTime[idx]>0:
                self.toleranceShells.append(toleranceShells[idx])
                self.toleranceTime.append(toleranceTime[idx])
        self.hbondAllShells = [self.bondLength]
        self.hbondAllShells.extend(self.toleranceShells)
        self.hbondAllShellsTime = [np.Inf]
        self.hbondAllShellsTime.extend(self.toleranceTime)
        self.cumsumhbondAllShells = np.cumsum(self.hbondAllShells)
        self.cumsumToleranceShells   = np.cumsum(self.toleranceShells)
        self.totalToleranceThichness = self.bondLength+sum(self.toleranceShells)
        # bondStartAtCore
        assert isinstance(bondStartAtCore, bool), Logger.error("bondStartAtCore must be boolean")
        self.bondStartAtCore = bondStartAtCore
        # bondStartWithinAngle
        assert isinstance(bondStartWithinAngle, bool), Logger.error("bondStartWithinAngle must be boolean")
        self.bondStartWithinAngle = bondStartWithinAngle
        # smoothHfJumps
        assert isinstance(smoothHfJumps, bool), Logger.error("smoothHfJumps must be boolean")
        self.smoothHfJumps = smoothHfJumps
        # set bin
        try:
            bin=float(bin)
        except:
            raise Logger.error("bin must be positive number")
        assert bin>0, Logger.error("bin must be non-zero positive number")
        assert bin<=1, Logger.error("bin must be smaller than 1 Angstrom")
        self.bin=bin
        self.bins = np.arange(0, self.totalToleranceThichness+self.bin, self.bin)
        # check selections
        self.elements = self._trajectory.elements
        for idx in self.donorsAtomsIndexes:
            if self.elements[idx].lower() not in ('o','n','f'):
                Logger.warn("donorsAtomsIndexes index '%s' is found to be '%s' instead of an oxygen 'o' or nitrogen 'n' or fluorine 'f'"%(idx, self.elements[idx].lower()))
        for idx in self.hydrogenAtomsIndexes:
            if self.elements[idx].lower()!="h":
                Logger.warn("hydrogenAtomsIndexes index '%s' is found to be '%s' instead of a hydrogen 'h'"%(idx, self.elements[idx].lower()))
        for idx in self.acceptorAtomsIndexes:
            if self.elements[idx].lower() not in ('o','n','f'):
                Logger.warn("acceptorAtomsIndexes index '%s' is found to be '%s' instead of an oxygen 'o' or nitrogen 'n' or fluorine 'f'"%(idx, self.elements[idx].lower()))
        # needed variables for analysis
        self.bondsDistances = -1*np.ones( (len(self.hydrogenAtomsIndexes), len(self.time)) ,dtype=np.float)
        self.bondsAngles    = -1*np.ones( (len(self.hydrogenAtomsIndexes), len(self.time)) ,dtype=np.float)
        self.acceptorsIndex = -1*np.ones( (len(self.hydrogenAtomsIndexes), len(self.time)) ,dtype=np.long)

    def __initialize_results__(self):
        # time
        self.results['time'] = np.array([self.time[idx] for idx in self.configurationsIndexes], dtype=np.float)
        # hbonds time
        self.results['hbonds_time'] =  np.zeros((len(self.configurationsIndexes)), dtype=np.float)
        # number of hBonds
        self.results['number_of_hbonds'] =  np.zeros((len(self.configurationsIndexes)), dtype=np.float)
        # hbonds time histogram
        self.results['histogram_bins'] = (self.bins[1:]+self.bins[0:-1])/2.0
        self.results['hbonds_distribution'] = 0.0*self.results['histogram_bins']
        # create histograms per shell
        self.shellsResultKeys = []
        shell = "0-->%sA_Infps"%(str(self.bondLength))
        self.shellsResultKeys.append(shell)
        self.results['hbonds_%s_time'%shell] =  np.zeros((len(self.configurationsIndexes)), dtype=np.float)
        self.results['hbonds_%s_distribution'%shell] = 0.0*self.results['histogram_bins']
        self.results['number_of_hbonds_%s'%shell] =  np.zeros((len(self.configurationsIndexes)), dtype=np.float)
        for idx in range(len(self.toleranceShells)):
            shell = "%s-->%sA_%sps"%(str(self.bondLength+self.cumsumToleranceShells[idx]-self.toleranceShells[idx]), str(self.bondLength+self.cumsumToleranceShells[idx]), str(self.toleranceTime[idx]))
            self.shellsResultKeys.append(shell)
            self.results['hbonds_%s_time'%shell] =  np.zeros((len(self.configurationsIndexes)), dtype=np.float)
            self.results['hbonds_%s_distribution'%shell] = 0.0*self.results['histogram_bins']
            self.results['number_of_hbonds_%s'%shell] =  np.zeros((len(self.configurationsIndexes)), dtype=np.float)

    def step(self, index):
        """
        analysis step of calculation method.\n

        :Parameters:
            #. atomIndex (int): the atom step index

        :Returns:
            #. stepData (object): object used in combine method
        """
        # get configuration index
        confIdx = self.configurationsIndexes[index]
        # set working configuration index
        self._trajectory.set_configuration_index(confIdx)
        # get coordinates
        coordinates = self._trajectory.boundaryConditions.real_to_box_array(self._trajectory.get_configuration_coordinates(confIdx), index=confIdx)
        # get box coordinates
        donorsBoxCoords   = self._trajectory.boundaryConditions.fold_box_array(coordinates[self.donorsAtomsIndexes])
        hydrogenBoxCoords = self._trajectory.boundaryConditions.fold_box_array(coordinates[self.hydrogenAtomsIndexes])
        acceptorBoxCoords = self._trajectory.boundaryConditions.fold_box_array(coordinates[self.acceptorAtomsIndexes])
        for hIdx in range(len(self.hydrogenAtomsIndexes)):
            diff = acceptorBoxCoords - hydrogenBoxCoords[hIdx]
            s = np.sign(diff)
            a = np.abs(diff)
            d = np.where(a < 0.5, a, 1-a)
            diff = (s*d).reshape((-1,3))
            diff = self._trajectory.boundaryConditions.box_to_real_array(diff, index=confIdx)
            dist = np.add.reduce( diff**2, axis = 1)
            # get accepted distances
            acceptedIndexes = np.where(dist <= self.totalToleranceThichness**2)[0]
            # check grouping
            acceptedIndexes = [idx for idx in acceptedIndexes if self.hydrogenAtomsGroup[hIdx]!=self.acceptorAtomsGroup[idx]]
            if not len(acceptedIndexes):
                continue
            acceptedRealDist = dist[acceptedIndexes]
            # append hbond distance and acceptor
            if len(acceptedIndexes) == 1:
                acceptedBondIndex = 0
            elif index == 0:
                acceptedBondIndex = np.argmin(acceptedRealDist)
            else:
                lastAcceptor = self.acceptorsIndex[hIdx, index-1]
                newAcceptors = [self.acceptorAtomsIndexes[idx] for idx in acceptedIndexes]
                if lastAcceptor in newAcceptors:
                    acceptedBondIndex = newAcceptors.index(lastAcceptor)
                else:
                    acceptedBondIndex = np.argmin(acceptedRealDist)
            # get donor to hydrogen vector
            diff = hydrogenBoxCoords[hIdx] - donorsBoxCoords[hIdx]
            s = np.sign(diff)
            a = np.abs(diff)
            d = np.where(a < 0.5, a, 1-a)
            diff = (s*d).reshape((-1,3))
            donorHydrogenVect = self._trajectory.boundaryConditions.box_to_real_array(diff, index=confIdx)[0]
            # get hydrogen to acceptor vector
            acceptorIndex = acceptedIndexes[acceptedBondIndex]
            diff = acceptorBoxCoords[acceptorIndex] -hydrogenBoxCoords[hIdx]
            s = np.sign(diff)
            a = np.abs(diff)
            d = np.where(a < 0.5, a, 1-a)
            diff = (s*d).reshape((-1,3))
            hydrogenAcceptorVect = self._trajectory.boundaryConditions.box_to_real_array(diff, index=confIdx)[0]
            # normalize vectors
            hydrogenAcceptorVect /= np.linalg.norm(hydrogenAcceptorVect)
            donorHydrogenVect    /= np.linalg.norm(donorHydrogenVect)
            # calculate angle
            angle = 180.-np.arccos( np.dot(hydrogenAcceptorVect, donorHydrogenVect ) )*180./np.pi
            # update arrays
            self.bondsAngles[hIdx, index]    = angle
            self.bondsDistances[hIdx, index] = np.sqrt( acceptedRealDist[acceptedBondIndex] )
            self.acceptorsIndex[hIdx, index] = self.acceptorAtomsIndexes[acceptedIndexes[acceptedBondIndex]]

        return index, None

    def combine(self, index, stepData):
        """
        analysis combine method called after each step.\n

        :Parameters:
            #. atomIndex (int): the atomIndex of the last calculated atom
            #. stepData (object): the returned data from step method
        """
        pass

    def finalize(self):
        """
        called once all the steps has been run.\n
        """
        # self.hbonds = [[[hIdx, shellIdx, shellTimeLimitsIdx, accIdx],[hIdx, shellIdx, shellTimeLimitsIdx,accIdx],...], ...  ]
        #                 ............first bond first shell..........,............first bond second shell2.......,....
        #                 ........................................first bond..........................................., .... second bond ...
        Logger.info("%s --> parsing and creating hydrogen bonds" %(self.__class__.__name__))
        self.hbonds = [[]]
        def create_hbonds():
            for bond in self.hbonds:
                if not len(bond):
                    continue
                # check all bond time threshold
                if self.time[bond[-1][2][1]]-self.time[bond[0][2][0]] < self.thresholdTime:
                    continue
                if self.bondStartAtCore and bond[0][1]:
                    continue
                self.results['hbonds_time'][bond[-1][2][1]-bond[0][2][0]] += 1
                self.results['number_of_hbonds'][bond[0][2][0]:bond[-1][2][1]] += 1
                for shell in bond:
                    hIdx = shell[0]
                    shellIdx = shell[1]
                    shellTimeIdx = shell[2]
                    shellKey = self.shellsResultKeys[shellIdx]
                    time = shellTimeIdx[1]-shellTimeIdx[0]
                    # append 1 to hbonds time
                    self.results['hbonds_%s_time'%shellKey][time] += 1
                    # update number of hbonds
                    self.results['number_of_hbonds_%s'%shellKey][shellTimeIdx[0]:shellTimeIdx[1]+1] += 1
                    # calculate bond lengths histogram
                    hist, _ = np.histogram(self.bondsDistances[hIdx, shellTimeIdx[0]:shellTimeIdx[1]+1], self.bins)
                    self.results['hbonds_%s_distribution'%shellKey] += hist


        def add_hbond(hIdx, shellIdx, shellTimeIdx, thisAcceptor, breakBond=False, debugMessage=""):
            if not (shellTimeIdx[0] is None) and not (shellTimeIdx[1] is None):
                #if not (self.time[shellTimeIdx[1]]-self.time[shellTimeIdx[0]]<=self.hbondAllShellsTime[shellIdx]):
                #    print debugMessage
                #    print "last shell index: %s"%str(lastShellIdx)
                #    print "this shell index: %s"%str(thisShellIdx)
                #    print "saving shell index: %s"%str(shellIdx)
                #    print "stayed between %s"%str(shellTimeIdx)
                #    print "total time of %s while allowed time is %s"%(str(self.time[timeIdx]-self.time[shellTimeIdx[0]]), str(self.hbondAllShellsTime[shellIdx]))
                #    print '\n'
                # first or new hbond added to the same hydrogen atom
                if not len(self.hbonds[-1]):
                    self.hbonds[-1].append([hIdx, shellIdx, shellTimeIdx, thisAcceptor])
                # new hydrogen index
                elif self.hbonds[-1][-1][0] != hIdx:
                    self.hbonds.append([[hIdx, shellIdx, shellTimeIdx, thisAcceptor]])
                # new acceptor
                elif self.hbonds[-1][-1][3] != thisAcceptor:
                    self.hbonds.append([[hIdx, shellIdx, shellTimeIdx, thisAcceptor]])
                # not continuous shellTimeIdx
                elif self.hbonds[-1][-1][2][1]+1 != shellTimeIdx[0]:
                    self.hbonds.append([[hIdx, shellIdx, shellTimeIdx, thisAcceptor]])
                else:
                    self.hbonds[-1].append([hIdx, shellIdx, shellTimeIdx, thisAcceptor])
                if breakBond:
                    self.hbonds.append([])

        def reset_shellTimeIdx_lowerAngleTime(thisDistance, thisAngle, thisTime):
            if self.bondStartAtCore and thisDistance>self.bondLength:
                shellTimeIdx = [None,None]
                lowerAngleTime = [None, None]
            elif self.bondStartWithinAngle and (thisAngle<self.bondAngleLimits[0] or thisAngle>self.bondAngleLimits[1]):
                shellTimeIdx = [None,None]
                lowerAngleTime = [None, None]
            else:
                shellTimeIdx = [timeIdx, timeIdx]
                # update bond angle time
                if thisAngle<self.bondAngleLimits[0] or thisAngle>self.bondAngleLimits[1]:
                    if lowerAngleTime[0] is None:
                        lowerAngleTime = [thisTime, thisTime]
                    else:
                        lowerAngleTime[1] =  thisTime
                else:
                    lowerAngleTime = [None, None]
            return  shellTimeIdx, lowerAngleTime

        # create hydrogen bonds
        for hIdx in range(len(self.hydrogenAtomsIndexes)):
            # create shellsIndexes list
            shellsIndexes = []
            for timeIdx in range(len(self.configurationsIndexes)):
                thisDistance = self.bondsDistances[hIdx, timeIdx]
                if thisDistance == -1:
                    shellsIndexes.append(-1)
                else:
                    indexes =[idx for idx in range(len(self.cumsumhbondAllShells)) if thisDistance <= self.cumsumhbondAllShells[idx]]
                    shellIdx = indexes[0]
                    shellsIndexes.append(shellIdx)
            # remove hfjumps
            if self.smoothHfJumps and len(shellsIndexes)>=3:
                for timeIdx in range(1,len(shellsIndexes)-1):
                    if self.bondsDistances[hIdx, timeIdx] == -1:
                        continue
                    if shellsIndexes[timeIdx-1] == shellsIndexes[timeIdx+1]:
                        shellsIndexes[timeIdx] = shellsIndexes[timeIdx-1]
                        self.bondsDistances[hIdx, timeIdx] = (self.bondsDistances[hIdx, timeIdx-1]+self.bondsDistances[hIdx, timeIdx+1])/2.
                    if self.acceptorsIndex[hIdx, timeIdx-1] == self.acceptorsIndex[hIdx, timeIdx+1]:
                        self.acceptorsIndex[hIdx, timeIdx] = self.acceptorsIndex[hIdx, timeIdx-1]
            # create hbonds
            shellTimeIdx = [None, None]
            lowerAngleTime = [None, None]
            for timeIdx in range(len(self.configurationsIndexes)):
                # get bond information
                thisShellIdx = shellsIndexes[timeIdx]
                thisAcceptor = self.acceptorsIndex[hIdx, timeIdx]
                thisAngle = self.bondsAngles[hIdx, timeIdx]
                thisTime = self.time[timeIdx]
                if timeIdx == 0:
                    lastShellIdx = thisShellIdx
                    lastAcceptor = thisAcceptor
                else:
                    lastAcceptor = self.acceptorsIndex[hIdx, timeIdx-1]
                    lastShellIdx = shellsIndexes[timeIdx-1]

                #################################################################################################
                ########################################## checking bond ########################################
                # check shell idx whether there is a bond or not
                if thisShellIdx == -1:
                    # add all valid bonds
                    add_hbond(hIdx, lastShellIdx, shellTimeIdx, thisAcceptor, breakBond=True, debugMessage="no bond found")
                    # reset parameters
                    shellTimeIdx = [None, None]
                    lowerAngleTime = [None, None]
                    continue

                # new bond or after non-bond thisShellIdx = -1
                if shellTimeIdx[0] is None:
                    shellTimeIdx, lowerAngleTime = reset_shellTimeIdx_lowerAngleTime(thisDistance, thisAngle, thisTime)
                    continue

                # check acceptors and break bond
                if thisAcceptor != lastAcceptor:
                    # add all valid bonds
                    add_hbond(hIdx, lastShellIdx, shellTimeIdx, thisAcceptor, breakBond=True, debugMessage="different acceptor")
                    # reset parameters
                    shellTimeIdx, lowerAngleTime = reset_shellTimeIdx_lowerAngleTime(thisDistance, thisAngle, thisTime)
                    continue

                # check bondAngle and break bond
                if thisAngle<self.bondAngleLimits[0] or thisAngle>self.bondAngleLimits[1]:
                    if (lowerAngleTime[0] is not None):
                        if (thisTime-lowerAngleTime[0]) > self.belowAngleToleranceTime:
                            # add all valid bonds
                            add_hbond(hIdx, lastShellIdx, shellTimeIdx, thisAcceptor, breakBond=True, debugMessage="angle time")
                            # reset parameters
                            shellTimeIdx = [None,None]
                            lowerAngleTime = [None, None]
                            continue
                        else:
                            lowerAngleTime[1] = thisTime
                    else:
                        lowerAngleTime = [thisTime, thisTime]
                else:
                    lowerAngleTime = [None, None]

                # Bond change shell
                if thisShellIdx != lastShellIdx:
                    # add all valid bonds
                    add_hbond(hIdx, lastShellIdx, shellTimeIdx, thisAcceptor, breakBond=False, debugMessage="jump to another shell")
                    # reset parameters
                    shellTimeIdx = [timeIdx, timeIdx]
                    continue

                # bond stay in same shell, check tolerance time
                elif thisTime-self.time[shellTimeIdx[0]]<=self.hbondAllShellsTime[thisShellIdx]:
                    shellTimeIdx[1] = timeIdx
                    continue

                # bond break because of tolerance time in outer shell
                else:
                    # add all valid bonds
                    add_hbond(hIdx, lastShellIdx, shellTimeIdx, thisAcceptor, breakBond=True, debugMessage="tolerance time violated")
                    # reset parameters
                    shellTimeIdx = [None,None]
                    lowerAngleTime = [None, None]
                    continue

            # final hbonds in hIdx
            add_hbond(hIdx, lastShellIdx, shellTimeIdx, thisAcceptor, breakBond=False)
        # create all registered hbond
        create_hbonds()

        # normalize and create the total
        for shellIdx in range(len(self.shellsResultKeys)):
            shellKey = self.shellsResultKeys[shellIdx]
            self.results['hbonds_%s_distribution'%shellKey] /= len(self.configurationsIndexes)
            self.results['hbonds_distribution'] += self.results['hbonds_%s_distribution'%shellKey]

        # Hydrogen bonds mean life time
        nonzero = list(np.where(self.results['hbonds_time'])[0])
        nHbonds = [self.results['hbonds_time'][idx] for idx in nonzero]
        times = [self.results['time'][idx] for idx in nonzero]
        times = [times[idx]*nHbonds[idx] for idx in range(len(times))]
        if len(times):
            self.results['hBond_mean_life_time'] = np.array([np.sum(times)])/len(times)
        else:
            self.results['hBond_mean_life_time'] = np.array([0.0])
