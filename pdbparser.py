"""
This is the main module of pdbparser package.
It contains the pdbparser class definition as well as pdbTrajectory.

.. inheritance-diagram:: pdbparser.pdbparser
    :parts: 2
"""
# This package is a personal effort and only developed according to my personal scientific needs.
# Any contribution and/or suggestion is more than welcome.

# standard libraries imports
from __future__ import print_function
import sys, copy, itertools, tempfile, os, shutil, re
try:
    import cPickle as pickle
except:
    import pickle
from numbers import Number

# external libraries imports
import numpy as np

# pdbparser library imports
from .log import Logger
from .Globals import PREFERENCES
from .Utilities.Geometry import *
from .Utilities.Information import *
from .Utilities.Modify import *
from .Utilities.Collection import *
from .Utilities.Database import *
from .Utilities.BoundaryConditions import InfiniteBoundaries, PeriodicBoundaries

# python version dependant imports
if int(sys.version[0])>=3:
    # THIS IS PYTHON 3
    str        = str
    long       = int
    unicode    = str
    bytes      = bytes
    basestring = str
    xrange     = range
    range      = lambda *args: list( xrange(*args) )
    maxint     = sys.maxsize
else:
    # THIS IS PYTHON 2
    str        = str
    unicode    = unicode
    bytes      = str
    long       = long
    basestring = basestring
    xrange     = xrange
    range      = range
    maxint     = sys.maxint

## get vmd path
def get_vmd_path():
    p = PREFERENCES.get('VMD_PATH', None)
    if p is not None:
        if sys.platform == "win32":
            p = r'"%s"'%p
        else:
            p = '%r'%p
    return p


def get_pymol_path():
    p = PREFERENCES.get('PYMOL_PATH', None)
    if p is not None:
        if sys.platform == "win32":
            p = r'"%s"'%p
        else:
            p = '%r'%p
    return p


def STR(s):
    if isinstance(s, str):
        return s
    elif isinstance(s, basestring):
        return str(s)
    elif isinstance(s, bytes):
        return str(s, 'utf-8', 'ignore')
    else:
        return str(s)

def _normalize_path(path):
    if os.sep=='\\':
        path = re.sub(r'([\\])\1+', r'\1', path).replace('\\','\\\\')
    return path

class pdbparser(object):
    """
    Initialize pdbparser instance

    :Parameters:
        #. filePath (None, string, list): the input pdb file path or list of
           pdb lines
    """
    __RECORD_NAMES__ = { "CRYST1" : "__read_CRYST1__", \
                         "ORIGX1" : "__read_ORIGXn__", \
                         "ORIGX2" : "__read_ORIGXn__", \
                         "ORIGX3" : "__read_ORIGXn__", \
                         "SCALE1" : "__read_SCALEn__", \
                         "SCALE2" : "__read_SCALEn__", \
                         "SCALE3" : "__read_SCALEn__", \
                         "MTRIXn" : "__read_MTRIXn__", \
                         "TVECT"  : "__read_TVECT__", \
                         "MODEL"  : "__read_MODEL__", \
                         "ATOM"   : "__read_ATOM__", \
                         "ANISOU" : "__read_ANISOU__", \
                         "TER"    : "__read_TER__", \
                         "HETATM" : "__read_HETATM__",\
                         "ENDMDL" : "__read_ENDMDL__",\
                         "END"    : "__read_END__" ,\
                        }

    def __init__(self, filePath = None):
        # initialize all attributes
        self.__reset__()
        # load pdb
        self.read_pdb(filePath)

    def _codify__(self, name='pdb', addDependencies=False, splitRecords=10):
        assert isinstance(splitRecords, int), Logger.error("splitRecords must be an integer")
        assert splitRecords>0, Logger.error("splitRecords must be >0")
        assert isinstance(name, basestring), Logger.error("name must be a string")
        assert re.match('[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None, Logger.error("given name '%s' can't be used as a variable name"%name)
        dependencies  = ['from pdbparser.pdbparser import pdbparser',
                         'from pdbparser.Utilities.BoundaryConditions import InfiniteBoundaries, PeriodicBoundaries']
        code          = ['{name} = pdbparser(filePath = None)'.format(name=name)]
        code.append( "{name}.set_name('{n}')".format(name=name, n=self.name) )
        if len(self.records):
            l = self.records
            n = splitRecords # stack splitRecords records per line
            if len(l)<=n:
                code.append( "{name}.records = {val}".format(name=name, val=self.records) )
            else:
                code.append( "{name}.records = []".format(name=name) )
                for s in [l[i * n:(i + 1) * n] for i in xrange(int((len(l) + n - 1)/n) )]:
                    code.append( "{name}.records.extend({val})".format(name=name, val=s) )
        if len(self.anisou):
            code.append( "{name}.anisou = {val}".format(name=name, val=self.anisou) )
        if len(self.models):
            code.append( "{name}.models = {val}".format(name=name, val=self.models) )
        if len(self.ter):
            code.append( "{name}.ter = {val}".format(name=name, val=self.ter) )
        if len(self.scalen):
            code.append( "{name}.scalen = {val}".format(name=name, val=self.scalen) )
        if self.tvect is not None:
            code.append( "{name}.tvect = '{val}'".format(name=name, val=self.tvect) )
        if len(self.headings):
            code.append( "{name}.headings = {val}".format(name=name, val=self.headings) )
        if len(self.connected):
            code.append( "{name}.connected = {val}".format(name=name, val=self.connected) )
        if len(self.origxn):
            code.append( "{name}.origxn = {val}".format(name=name, val=self.origxn) )
        if len(self.connected):
            code.append( "{name}.connected = {val}".format(name=name, val=self.connected) )
        ## if len(self.crystallographicStructure): ## SET IN set_boundary_conditions
        ##     code.append( "{name}.crystallographicStructure = val".format(name=name, val=self.crystallographicStructure) )
        if isinstance(self._boundaryConditions, PeriodicBoundaries):
            vectors = [list(i) for i in list(self._boundaryConditions.get_vectors())]
            code.append( "bc = PeriodicBoundaries()" )
            code.append( "bc.set_vectors({vectors})".format(vectors=vectors) )
        else:
            code.append( "bc = InfiniteBoundaries()")
        code.append( "{name}.set_boundary_conditions(bc)".format(name=name))
        # return
        return dependencies, '\n'.join(code)

    def __reset__(self):
        self.filePath=None
        self.__name = None
        self.records = []
        self.anisou = {}
        self.models = {}
        self.ter = {}
        self.scalen = []
        self.tvect = None
        self.headings = []
        self.connected = []
        # origin
        self.origxn = []
        self.origxn.append( {'record_name': 'ORIGX1',\
                             'o[n][1]': 1.0 , 'o[n][2]': 0.0 , 'o[n][3]': 0.0,  't[n]': 0.0} )
        self.origxn.append( {'record_name': 'ORIGX2',\
                             'o[n][1]': 0.0 , 'o[n][2]': 1.0 , 'o[n][3]': 0.0,  't[n]': 0.0} )
        self.origxn.append( {'record_name': 'ORIGX3',\
                             'o[n][1]': 0.0 , 'o[n][2]': 0.0 , 'o[n][3]': 1.0,  't[n]': 0.0} )
        # crystallography
        self.crystallographicStructure = { "record_name": "CRYST1" ,\
                                           "a"          : float( 1.000 ) ,\
                                           "b"          : float( 1.000 ) ,\
                                           "c"          : float( 1.000 ) ,\
                                           "alpha"      : float( 90.00 ) ,\
                                           "beta"       : float( 90.00 ) ,\
                                           "gamma"      : float( 90.00 ) ,\
                                           "space_group": "P 1       " ,\
                                           "z_value"    : 1 ,\
                                          }
        # boundary conditions
        self._boundaryConditions = InfiniteBoundaries()

    def __getitem__(self, index):
        return self.records[index]

    def __getslice__(self, i, j, step = 1):
        return self.records[max(0, i):max(0, j):step]

    def __len__(self):
        return len(self.records)

    @property
    def name(self):
        """ Get the pdb setted name. """
        return self.__name

    @property
    def indexes(self):
        """ Get a list of all indexes. identical to range(numberOfRecords). """
        return range( len(self.records) )

    @property
    def atomsIndexes(self):
        """ Get a list of all indexes. identical to range(numberOfRecords). """
        return range( len(self.records) )

    @property
    def xatomsIndexes(self):
        """ Gets a generator of all indexes . identical to xrange(numberOfRecords)."""
        return xrange( len(self.records) )

    @property
    def xindexes(self):
        """ alias to xatomsIndexes. """
        return xrange( len(self.records) )

    @property
    def coordinates(self):
        """ Get all atoms coordinates as a numpy.array(N,3)."""
        return get_coordinates(self.indexes, self)

    @property
    def elements(self):
        """ Get a list of all atoms elements. """
        return get_records_attribute_values(self.indexes, self, "element_symbol")

    @property
    def names(self):
        """ Get a list of all atoms name. """
        return get_records_attribute_values(self.indexes, self, "atom_name")

    @property
    def residues(self):
        """ Get a list of all atoms residue name.  """
        return get_records_attribute_values(self.indexes, self, "residue_name")

    @property
    def sequences(self):
        """ Get a list of all atoms sequence number. """
        return get_records_attribute_values(self.indexes, self, "sequence_number")

    @property
    def segments(self):
        """ Get a list of all atoms segment identifier. """
        return get_records_attribute_values(self.indexes, self, "segment_identifier")

    @property
    def chainIdentifier(self):
        """ Get a list of all atoms chain identifier. """
        return get_records_attribute_values(self.indexes, self, "chain_identifier")

    @property
    def numberOfConfigurations(self):
        """ Get the number of configurations. It will always return 1 because it's a pdbparser instance. """
        return 1

    @property
    def numberOfAtoms(self):
        """ alias to len(), returns the number of atoms in the pdb. """
        return len(self)

    @property
    def boundaryConditions(self):
        """ Get the boundary condition instance. """
        return self._boundaryConditions

    @property
    def simulationBox(self):
        """ alias to boundaryConditions """
        return self._boundaryConditions

    def __read_TVECT__(self, line, model, index):
        """
        Contains: the translation vector which have infinite covalent connections
        Notes: For structures not comprised of discrete molecules (e.g., infinite
        polysaccharide chains), the entry contains a fragment which can be built into
        the full structure by the simple translation vectors of TVECT records.
        COLUMNS       DATA TYPE      CONTENTS
        --------------------------------------------------------------------------------
         1 -  6       Record name    "TVECT "
         8 - 10       Integer        Serial number
        11 - 20       Real(10.5)     t[1]
        21 - 30       Real(10.5)     t[2]
        31 - 40       Real(10.5)     t[3]
        41 - 70       String         Text comment

        Example:
                 1         2         3         4         5         6         7
        1234567890123456789012345678901234567890123456789012345678901234567890
        TVECT    1   0.00000   0.00000  28.30000
        """
        # not defined yet
        self.tvect = None
        return model

    def __read_ANISOU__(self, line, model, index):
        """
        Contains: the anisotropic temperature factors
        Notes:
        Columns 7 - 27 and 73 - 80 are identical to the corresponding ATOM/HETATM record.
        The anisotropic temperature factors (columns 29 - 70) are scaled by a factor
        of 10**4 (Angstroms**2) and are presented as integers.
        The anisotropic temperature factors are stored in the same coordinate frame as the
        atomic coordinate records.
        ANISOU values are listed only if they have been provided by the depositor.

        COLUMNS        DATA TYPE       CONTENTS
        ----------------------------------------------------------------------
        1 -  6        Record name     "ANISOU"
        7 - 11        Integer         Atom serial number
        13 - 16       Atom            Atom name
        17            Character       Alternate location indicator
        18 - 20       Residue name    Residue name
        22            Character       Chain identifier
        23 - 26       Integer         Residue sequence number
        27            AChar           Insertion code
        29 - 35       Integer         u[1][1]
        36 - 42       Integer         u[2][2]
        43 - 49       Integer         u[3][3]
        50 - 56       Integer         u[1][2]
        57 - 63       Integer         u[1][3]
        64 - 70       Integer         u[2][3]
        73 - 76       LString(4)      Segment identifier, left-justified
        77 - 78       LString(2)      Element symbol, right-justified
        79 - 80       LString(2)      Charge on the atom

        Example:
                 1         2         3         4         5         6         7         8
        12345678901234567890123456789012345678901234567890123456789012345678901234567890
        ATOM    107  N   GLY    13      12.681  37.302 -25.211 1.000 15.56           N
        ANISOU  107  N   GLY    13     2406   1892   1614    198    519   -328       N
        ATOM    108  CA  GLY    13      11.982  37.996 -26.241 1.000 16.92           C
        ANISOU  108  CA  GLY    13     2748   2004   1679    -21    155   -419       C
        ATOM    109  C   GLY    13      11.678  39.447 -26.008 1.000 15.73           C
        ANISOU  109  C   GLY    13     2555   1955   1468     87    357   -109       C
        ATOM    110  O   GLY    13      11.444  40.201 -26.971 1.000 20.93           O
        ANISOU  110  O   GLY    13     3837   2505   1611    164   -121    189       O
        ATOM    111  N   ASN    14      11.608  39.863 -24.755 1.000 13.68           N
        ANISOU  111  N   ASN    14     2059   1674   1462     27    244    -96       N
        """
        try:
            self.anisou[ len(self.records)-1 ] = { "record_name"        : STR( line[0:6] ).strip() ,\
                                                   "serial_number"      : INT( line[6:11] ) ,\
                                                   "atom_name"          : STR( line[12:16] ).strip() ,\
                                                   "location_indicator" : STR( line[16] ).strip() ,\
                                                   "residue_name"       : STR( line[17:20] ).strip() ,\
                                                   "chain_identifier"   : STR( line[21] ).strip() ,\
                                                   "sequence_number"    : INT( line[22:26] ) ,\
                                                   "code_of_insertion"  : STR( line[26] ).strip() ,\
                                                   "u[1][1]"            : INT( line[28:35] ) ,\
                                                   "u[2][2]"            : INT( line[35:42] ) ,\
                                                   "u[3][3]"            : INT( line[42:49] ) ,\
                                                   "u[1][2]"            : INT( line[49:56] ) ,\
                                                   "u[1][3]"            : INT( line[56:63] ) ,\
                                                   "u[2][3]"            : INT( line[63:70] ) ,\
                                                   "segment_identifier" : STR( line[72:76] ).strip() ,\
                                                   "element_symbol"     : STR( line[76:78] ).strip() ,\
                                                   "charge"             : STR( line[78:80] ).strip() ,\
                                                  }
        except Exception as err:
            Logger.error("Unable to read line number '{i}' for ANISOU '{l}'".format(l=line.replace('\n',''),i=index))
        # return model
        return model

    def __read_SCALEn__(self, line, model, index):
        """
        Contains: the transformation from the orthogonal coordinates contained in
        the entry to fractional crystallographic coordinates
        Notes:
        If the orthogonal Angstroms coordinates are X, Y, Z, and the fractional
        cell coordinates are xfrac, yfrac, zfrac, then:
        xfrac = S11X + S12Y + S13Z + U1
        yfrac = S21X + S22Y + S23Z + U2
        zfrac = S31X + S32Y + S33Z + U3
        For NMR and fiber diffraction submissions, SCALE is given as an identity
        matrix with no translation.
       COLUMNS       DATA TYPE      CONTENTS
       --------------------------------------------------------------------------------
        1 -  6       Record name    "SCALEn" (n=1, 2, or 3)
        11 - 20       Real(10.6)     s[n][1]
        21 - 30       Real(10.6)     s[n][2]
        31 - 40       Real(10.6)     s[n][3]
        46 - 55       Real(10.5)     u[n]

        Example:
                 1         2         3         4         5         6         7
        1234567890123456789012345678901234567890123456789012345678901234567890
        SCALE1      0.019231  0.000000  0.000000        0.00000
        SCALE2      0.000000  0.017065  0.000000        0.00000
        SCALE3      0.000000  0.000000  0.016155        0.00000
        """
        try:
            self.scalen.append( { "record_name": STR( line[0:6] ).strip() ,\
                                  "s[n][1]"    : FLOAT( line[10:20] ) ,\
                                  "s[n][2]"    : FLOAT( line[20:30] ) ,\
                                  "s[n][3]"    : FLOAT( line[30:40] ) ,\
                                  "u[n]"       : FLOAT( line[45:55] ) ,\
                                } )
        except Exception as err:
            Logger.error("Unable to read line number '{i}' for SCALE '{l}'".format(l=line.replace('\n',''),i=index))
        # return model
        return model

    def __read_ORIGXn__(self, line, model, index):
        """
        Contains: the transformation from the orthogonal coordinates contained
        in the database entry to the submitted coordinates
        Notes: If the original submitted coordinates are Xsub, Ysub, Zsub and the
        orthogonal Angstroms coordinates contained in the data entry are X, Y, Z, then:
        Xsub = O11X + O12Y + O13Z + T1
        Ysub = O21X + O22Y + O23Z + T2
        Zsub = O31X + O32Y + O33Z + T3
        COLUMNS       DATA TYPE       CONTENTS
        --------------------------------------------------------------------------------
         1 -  6       Record name     "ORIGXn" (n=1, 2, or 3)
        11 - 20       Real(10.6)      o[n][1]
        21 - 30       Real(10.6)      o[n][2]
        31 - 40       Real(10.6)      o[n][3]
        46 - 55       Real(10.5)      t[n]

        Example:

                 1         2         3         4         5         6         7
        1234567890123456789012345678901234567890123456789012345678901234567890
        ORIGX1      0.963457  0.136613  0.230424       16.61000
        ORIGX2     -0.158977  0.983924  0.081383       13.72000
        ORIGX3     -0.215598 -0.115048  0.969683       37.65000
        """
        if len(self.origxn) == 3:
            self.origxn = []
        try:
            self.origxn.append( { "record_name": STR( line[0:6] ).strip() ,\
                                  "o[n][1]"    : FLOAT( line[10:20] ) ,\
                                  "o[n][2]"    : FLOAT( line[20:30] ) ,\
                                  "o[n][3]"    : FLOAT( line[30:40] ) ,\
                                  "t[n]"       : FLOAT( line[45:55] ) ,\
                                } )
        except Exception as err:
            Logger.error("Unable to read line number '{i}' for ORIGX '{l}'".format(l=line.replace('\n',''),i=index))
        # return model
        return model

    def __read_MODEL__(self, line, model, index):
        """
        Contains: the model serial number when a single coordinate entry contains
        multiple structures
        Notes:
        Models are numbered sequentially beginning with 1.
        If an entry contains more than 99,999 total atoms,
        then it must be divided among multiple models.
        Each MODEL must have a corresponding ENDMDL record.
        In the case of an NMR entry the EXPDTA record states the number of model
        structures that are present in the individual entry.

        COLUMNS       DATA TYPE      CONTENTS
        ----------------------------------------------------------------------
         1 -  6       Record name    "MODEL "
         11 - 14       Integer        Model serial number

         Example:
                 1         2         3         4         5         6         7         8
        12345678901234567890123456789012345678901234567890123456789012345678901234567890
        MODEL        1
        ATOM      1  N   ALA     1      11.104   6.134  -6.504  1.00  0.00           N
        ATOM      2  CA  ALA     1      11.639   6.071  -5.147  1.00  0.00           C
        ...
        ...
        ATOM    293 1HG  GLU    18     -14.861  -4.847   0.361  1.00  0.00           H
        ATOM    294 2HG  GLU    18     -13.518  -3.769   0.084  1.00  0.00           H
        TER     295      GLU    18
        ENDMDL
        MODEL        2
        ATOM      1  N   ALA     1      11.304   6.234  -6.104  1.00  0.00           N
        ATOM      2  CA  ALA     1      11.239   6.371  -5.247  1.00  0.00           C
        ...
        ...
        ATOM    293 1HG  GLU    18     -14.752  -4.948   0.461  1.00  0.00           H
        ATOM    294 2HG  GLU    18     -13.630  -3.769   0.160  1.00  0.00           H
        TER     295      GLU    18
        ENDMDL
        """
        try:
            model = { "record_name"         : STR( line[0:6] ).strip()  ,\
                      "model_serial_number" : INT( line[10:14] ) ,\
                      "model_start"         : len(self.records) ,\
                      "model_end"           : None ,\
                      "termodel"            : None ,\
                      "endmodel"            : None ,\
                      "MODEL_NAME"          : self.define_model_name()
                     }
        except Exception as err:
            Logger.error("Unable to read line number '{i}' for MODEL '{l}'".format(l=line.replace('\n',''),i=index))
        # return model
        return model

    def __read_ENDMDL__(self, line, model, index):
        """
        Contains: these records are paired with MODEL records to group individual
        structures found in a coordinate entry
        Notes:
        MODEL/ENDMDL records are used only when more than one structure
        is presented in the entry, or if there are more than 99,999 atoms.
        Every MODEL record has an associated ENDMDL record.
        COLUMNS         DATA TYPE        CONTENTS
        ------------------------------------------------------------------
        1 -  6         Record name      "ENDMDL"

        Example:

                 1         2         3         4         5         6         7         8
        12345678901234567890123456789012345678901234567890123456789012345678901234567890
        ...
        ...
        ATOM  14550 1HG  GLU   122     -14.364  14.787 -14.258  1.00  0.00           H
        ATOM  14551 2HG  GLU   122     -13.794  13.738 -12.961  1.00  0.00           H
        TER   14552      GLU   122
        ENDMDL
        MODEL        9
        ATOM  14553  N   SER     1     -28.280   1.567  12.004  1.00  0.00           N
        ATOM  14554  CA  SER     1     -27.749   0.392  11.256  1.00  0.00           C
        ...
        ...
        ATOM  16369 1HG  GLU   122      -3.757  18.546  -8.439  1.00  0.00           H
        ATOM  16370 2HG  GLU   122      -3.066  17.166  -7.584  1.00  0.00           H
        TER   16371      GLU   122
        ENDMDL
        MODEL       10
        ATOM  16372  N   SER     1     -22.285   7.041  10.003  1.00  0.00           N
        ATOM  16373  CA  SER     1     -23.026   6.872   8.720  1.00  0.00           C
        ...
        ...
        ATOM  18188 1HG  GLU   122      -1.467  18.282 -17.144  1.00  0.00           H
        ATOM  18189 2HG  GLU   122      -2.711  18.067 -15.913  1.00  0.00           H
        TER   18190      GLU   122
        ENDMDL
        """
        try:
            model["model_end"] = len(self.records)
            model["endmodel"]  =  STR( line[0:6] ).strip()
        except Exception as err:
            Logger.error("Unable to read line number '{i}' for ENDMDL '{l}'".format(l=line.replace('\n',''),i=index))
        self.models[model["model_serial_number"]] = model
        return None

    def __read_CRYST1__(self, line, model, index):
        """
        Contains: unit cell parameters, space group, and Z value
        Notes:
        If the structure was not determined by crystallographic means, simply defines
        a unit cube (a = b =c = 1.0, alpha = beta = gamma = 90
        degrees, space group = P 1, and Z = 1)
        The Hermann-Mauguin space group symbol is given without parenthesis,
        e.g., P 21 21 2 and using the full symbol, e.g., C 1 2 1 instead of C 2.
        The screw axis is described as a two digit number.
        For a rhombohedral space group in the hexagonal setting, the lattice type symbol
        used is H. The Z value is the number of polymeric chains in a unit cell.
        In the case of heteropolymers, Z is the number of occurrences of the most
        populous chain. In the case of a polycrystalline fiber diffraction study,
        CRYST1 and SCALE contain the normal unit cell data.
        The unit cell parameters are used to calculate SCALE.

        COLUMNS       DATA TYPE      CONTENTS
        --------------------------------------------------------------------------------
         1 -  6       Record name    "CRYST1"
         7 - 15       Real(9.3)      a (Angstroms)
        16 - 24       Real(9.3)      b (Angstroms)
        25 - 33       Real(9.3)      c (Angstroms)
        34 - 40       Real(7.2)      alpha (degrees)
        41 - 47       Real(7.2)      beta (degrees)
        48 - 54       Real(7.2)      gamma (degrees)
        56 - 66       LString        Space group
        67 - 70       Integer        Z value

        Example:
                 1         2         3         4         5         6         7
        1234567890123456789012345678901234567890123456789012345678901234567890
        CRYST1  117.000   15.000   39.000  90.00  90.00  90.00 P 21 21 21    8
        """
        try:
            self.crystallographicStructure = { "record_name": STR( line[0:6] ).strip() ,\
                                               "a"          : FLOAT( line[6:15] ) ,\
                                               "b"          : FLOAT( line[15:24] ) ,\
                                               "c"          : FLOAT( line[24:33] ) ,\
                                               "alpha"      : FLOAT( line[33:40] ) ,\
                                               "beta"       : FLOAT( line[40:47] ) ,\
                                               "gamma"      : FLOAT( line[47:54] ) ,\
                                               "space_group": STR( line[55:66] ).strip() ,\
                                               "z_value"    : INT( line[66:70] ) ,\
                                             }
        except Exception as err:
            Logger.error("Unable to read line number '{i}' for CRYST '{l}'".format(l=line.replace('\n',''),i=index))
        # return model
        return model

    def  __read_TER__(self, line, model, index):
        """
        Contains: indicates the end of a list of ATOM/HETATM records for a chain
        Notes:
        The TER records occur in the coordinate section of the entry, and indicate
        the last residue presented for each polypeptide and/or nucleic acid chain for
        which there are coordinates. For proteins, the residue defined on the TER
        record is the carboxy-terminal residue; for nucleic acids it is the
        3'-terminal residue.
        For a cyclic molecule, the choice of termini is arbitrary.
        Terminal oxygen atoms are presented as OXT for proteins, and as O5T or O3T for
        nucleic acids. The TER record has the same residue name, chain identifier,
        sequence number and insertion code as the terminal residue. The serial number
        of the TER record is one number greater than the serial number of the ATOM/HETATM
        preceding the TER.
        For chains with gaps due to disorder, it is recommended that the C-terminus
        atoms be labeled O and OXT.
        The residue name appearing on the TER record must be the same as the residue name
        of the immediately preceding ATOM or non-water HETATM record.

        COLUMNS         DATA TYPE         CONTENTS
        --------------------------------------------------------------------------------
        1 -  6        Record name       "TER   "
        7 - 11        Integer           Serial number
        18 - 20       Residue name      Residue name
        22            Character         Chain identifier
        23 - 26       Integer           Residue sequence number
        27            AChar             Insertion code

        Example:

                 1         2         3         4         5         6         7         8
        12345678901234567890123456789012345678901234567890123456789012345678901234567890
        ATOM   4150  H   ALA A 431       8.674  16.036  12.858  1.00  0.00           H
        TER    4151      ALA A 431

        ATOM   1403  O   PRO P  22      12.701  33.564  15.827  1.09 18.03           O
        ATOM   1404  CB  PRO P  22      13.512  32.617  18.642  1.09  9.32           C
        ATOM   1405  CG  PRO P  22      12.828  33.382  19.740  1.09 12.23           C
        ATOM   1406  CD  PRO P  22      12.324  34.603  18.985  1.09 11.47           C
        HETATM 1407  CA  BLE P   1      14.625  32.240  14.151  1.09 16.76           C
        HETATM 1408  CB  BLE P   1      15.610  33.091  13.297  1.09 16.56           C
        HETATM 1409  CG  BLE P   1      15.558  34.629  13.373  1.09 14.27           C
        HETATM 1410  CD1 BLE P   1      16.601  35.208  12.440  1.09 14.75           C
        HETATM 1411  CD2 BLE P   1      14.209  35.160  12.930  1.09 15.60           C
        HETATM 1412  N   BLE P   1      14.777  32.703  15.531  1.09 14.79           N
        HETATM 1413  B   BLE P   1      14.921  30.655  14.194  1.09 15.56           B
        HETATM 1414  O1  BLE P   1      14.852  30.178  12.832  1.09 16.10           O
        HETATM 1415  O2  BLE P   1      13.775  30.147  14.862  1.09 20.95           O
        TER    1416      BLE P   1
        """
        try:
            ter =  { "record_name"       : STR( line[0:6] ).strip() ,\
                     "serial_number"     : INT( line[6:11] ) ,\
                     "residue_name"      : STR( line[17:20] ).strip()  ,\
                     "chain_identifier"  : STR( line[21] ).strip() ,\
                     "sequence_number"   : INT( line[22:26] ) ,\
                     "code_of_insertion" : STR( line[26] ).strip() ,\
                     "INDEX_IN_RECORDS"  : len(self.records) ,\
                   }
        except Exception as err:
            Logger.error("Unable to read line number '{i}' for TER '{l}'".format(l=line.replace('\n',''),i=index))
        else:
            if model is not None:
                if model['termodel'] is not None:
                    self.ter[ copy.deepcopy(model["termodel"]['INDEX_IN_RECORDS']) ] = copy.deepcopy(model["termodel"] )
                model["termodel"] = ter
            else:
                self.ter[len(self.records)] = ter
        # return model
        return model

    def  __read_END__(self, line, model, index):
        """
        this indicates the end of the pdb file
        """
        return model

    def __read_HETATM__(self, line, model, index):
        """
        Contains: the atomic coordinate records for atoms within "non-standard"
        groups. These records are used for water molecules and atoms presented in HET
        groups.
        Notes:
        Insertion codes, segment id, and element naming are fully described in the
        ATOM section of this document.
        Disordered solvents may be represented by the residue name DIS.
        No ordering is specified for polysaccharides.
        HETATM records must have corresponding HET, HETNAM, FORMUL
        and CONECT records, except for waters.

        COLUMNS        DATA TYPE       CONTENTS
        --------------------------------------------------------------------------------
        1 -  6        Record name     "HETATM"
        7 - 11        Integer         Atom serial number.
        13 - 16       Atom            Atom name
        17            Character       Alternate location indicator
        18 - 20       Residue name    Residue name
        22            Character       Chain identifier
        23 - 26       Integer         Residue sequence number
        27            AChar           Code for insertion of residues
        31 - 38       Real(8.3)       Orthogonal coordinates for X
        39 - 46       Real(8.3)       Orthogonal coordinates for Y
        47 - 54       Real(8.3)       Orthogonal coordinates for Z
        55 - 60       Real(6.2)       Occupancy
        61 - 66       Real(6.2)       Temperature factor
        73 - 76       LString(4)      Segment identifier, left-justified
        77 - 78       LString(2)      Element symbol, right-justified
        79 - 80       LString(2)      Charge on the atom

        Example:
                 1         2         3         4         5         6         7         8
        12345678901234567890123456789012345678901234567890123456789012345678901234567890
        HETATM 1357 MG    MG   168       4.669  34.118  19.123  1.00  3.16          MG2+
        HETATM 3835 FE   HEM     1      17.140   3.115  15.066  1.00 14.14          FE3+
        """
        # it is the same format as ATOM record
        model = self.__read_ATOM__(line, model)
        # return model
        return model

    def __read_ATOM__(self, line, model, index):
        """
        Contains: the atomic coordinates for standard residues and the occupancy and
        temperature factor for each atom
        Notes:
        ATOM records for proteins are listed from amino to carboxyl terminus.
        Nucleic acid residues are listed from the 5' to the 3' terminus.
        No ordering is specified for polysaccharides.
        The list of ATOM records in a chain is terminated by a TER record.
        If an atom is provided in more than one position, then a non-blank alternate
        location indicator must be used. Within a residue, all atoms of a given
        conformation are assigned the same alternate position indicator.
        Additional atoms (modifying group) to side chains of standard residues are
        represented as a HET group which is assigned its own residue name. The chainID,
        sequence number, and insertion code assigned to the HET group is that of the
        standard residue to which it is attached.
        In some entries, the occupancy and temperature factor fields may be used
        for other quantities. The segment identifier is a string of up to four (4)
        alphanumeric characters, left-justified, and may include a space, e.g.,
        CH86, A 1, NASE.

        COLUMNS        DATA TYPE       CONTENTS
        --------------------------------------------------------------------------------
        1 -  6        Record name     "ATOM  "
        7 - 11        Integer         Atom serial number.
        13 - 16       Atom            Atom name.
        17            Character       Alternate location indicator.
        18 - 20       Residue name    Residue name.
        22            Character       Chain identifier.
        23 - 26       Integer         Residue sequence number.
        27            AChar           Code for insertion of residues.
        31 - 38       Real(8.3)       Orthogonal coordinates for X in Angstroms.
        39 - 46       Real(8.3)       Orthogonal coordinates for Y in Angstroms.
        47 - 54       Real(8.3)       Orthogonal coordinates for Z in Angstroms.
        55 - 60       Real(6.2)       Occupancy.
        61 - 66       Real(6.2)       Temperature factor (Default = 0.0).
        73 - 76       LString(4)      Segment identifier, left-justified.
        77 - 78       LString(2)      Element symbol, right-justified.
        79 - 80       LString(2)      Charge on the atom.

        Example:
                 1         2         3         4         5         6         7         8
        12345678901234567890123456789012345678901234567890123456789012345678901234567890
        ATOM    145  N   VAL A  25      32.433  16.336  57.540  1.00 11.92      A1   N
        ATOM    146  CA  VAL A  25      31.132  16.439  58.160  1.00 11.85      A1   C
        ATOM    147  C   VAL A  25      30.447  15.105  58.363  1.00 12.34      A1   C
        ATOM    148  O   VAL A  25      29.520  15.059  59.174  1.00 15.65      A1   O
        ATOM    149  CB AVAL A  25      30.385  17.437  57.230  0.28 13.88      A1   C
        ATOM    150  CB BVAL A  25      30.166  17.399  57.373  0.72 15.41      A1   C
        ATOM    151  CG1AVAL A  25      28.870  17.401  57.336  0.28 12.64      A1   C
        ATOM    152  CG1BVAL A  25      30.805  18.788  57.449  0.72 15.11      A1   C
        ATOM    153  CG2AVAL A  25      30.835  18.826  57.661  0.28 13.58      A1   C
        ATOM    154  CG2BVAL A  25      29.909  16.996  55.922  0.72 13.25      A1   C
        """
        try:
            self.records.append( { "record_name"       : STR( line[0:6] ).strip() ,\
                                   "serial_number"     : INT( line[6:11] ) ,\
                                   "atom_name"         : STR( line[12:16] ).strip() ,\
                                   "location_indicator": STR( line[16] ).strip()  ,\
                                   "residue_name"      : STR( line[17:20] ).strip()  ,\
                                   "chain_identifier"  : STR( line[21] ).strip()  ,\
                                   "sequence_number"   : INT( line[22:26] ) ,\
                                   "code_of_insertion" : STR( line[26] ).strip()  ,\
                                   "coordinates_x"     : FLOAT( line[30:38] ) ,\
                                   "coordinates_y"     : FLOAT( line[38:46] ) ,\
                                   "coordinates_z"     : FLOAT( line[46:54] ) ,\
                                   "occupancy"         : FLOAT( line[54:60] ) ,\
                                   "temperature_factor": FLOAT( line[60:66] ) ,\
                                   "segment_identifier": STR( line[72:76] ).strip()  ,\
                                   "element_symbol"    : STR( line[76:78] ).strip()  ,\
                                   "charge"            : STR( line[78:80] ).strip()  ,\
                                  } )
        except Exception as err:
            Logger.error("Unable to read line number '{i}' for ATOM '{l}'".format(l=line.replace('\n',''),i=index))
        # return model
        return model

    def __write_pdb(self, fd ,\
                    headings = True ,\
                    additionalRemarks = None,\
                    structure = True ,\
                    model_format = False ,\
                    ter_format = True ,\
                    origxn = True ,\
                    scalen = True ,\
                    anisou = False,
                    coordinates=coordinates,
                    boundaryConditions = boundaryConditions):
        # write pdbparser header
        fd.write('REMARK    this file is generated using %r package' %self.__class__.__name__)
        fd.write('\n')
        # write pdbparser boundaryConditions
        bc=None
        if boundaryConditions is None:
            if hasattr(self, "_boundaryConditions"):
                if isinstance(self._boundaryConditions, PeriodicBoundaries):
                    bc = self._boundaryConditions
        elif isinstance(boundaryConditions, PeriodicBoundaries):
            bc = boundaryConditions
        else:
            assert isinstance(boundaryConditions, InfiniteBoundaries),"boundaryConditions must be None or either InfiniteBoundaries and PeriodicBoundaries."
        if bc is not None:
            v = bc.get_vectors()
            vectors = "%s  %s  %s  %s  %s  %s  %s  %s  %s"%(STR(v[0,0]),STR(v[0,1]),STR(v[0,2]),STR(v[1,0]),STR(v[1,1]),STR(v[1,2]),STR(v[2,0]),STR(v[2,1]),STR(v[2,2]))
            fd.write("REMARK    Boundary Conditions: %s \n" %vectors)
        # write headings
        if headings:
            for line in self.headings:
                line = line.strip()
                if not line:
                    continue
                if 'REMARK    this file is generated using %r package' %self.__class__.__name__ in line:
                    continue
                fd.write(line)
        # write additional remarks
        if additionalRemarks is not None:
            if isinstance(additionalRemarks, str):
                additionalRemarks = [additionalRemarks]
            else:
                assert isinstance(additionalRemarks, (list, tuple))
            for remark in additionalRemarks:
                fd.write('REMARK    %s' %remark)
        # write structure
        if structure:
            structure = STR('%s'%self.crystallographicStructure['record_name']).ljust(6, " ")[:6]  +\
                        STR('%9.3f'%self.crystallographicStructure['a']).rjust(9, " ")[-9:]  + STR('%9.3f'%self.crystallographicStructure['b']).rjust(9, " ")[-9:]  + STR('%9.3f'%self.crystallographicStructure['c']).rjust(9, " ")[-9:] +\
                        STR('%7.2f'%self.crystallographicStructure['alpha']).rjust(7, " ")[-7:] + STR('%7.2f'%self.crystallographicStructure['beta']).rjust(7, " ")[-7:] + STR('%7.2f'%self.crystallographicStructure['gamma']).rjust(7, " ")[-7:]+\
                        STR(' ') + STR('%s'%self.crystallographicStructure['space_group']).ljust(11, " ")[:11] + STR('%i'%self.crystallographicStructure['z_value']).rjust(4, " ")[-4:]
            fd.write(structure)
            fd.write('\n')
        # write origxn
        if origxn:
            for orig in self.origxn:
                origLine = '%s'%STR(orig['record_name']).ljust(6, " ")[:6]  +\
                           '    ' +\
                           STR('%10.6f'%orig['o[n][1]']).rjust(10, " ")[-10:]  + STR('%10.6f'%orig['o[n][2]']).rjust(10, " ")[-10:]  + STR('%10.6f'%orig['o[n][3]']).rjust(10," ")[-10:] +\
                           '     ' +\
                           STR('%10.5f'%orig['t[n]']).rjust(10, " ")[-10:]
                fd.write(origLine)
                fd.write('\n')
        # write scalen
        if scalen:
            for scale in self.scalen:
                origLine = '%s'%STR(scale['record_name']).ljust(6, " ")[:6]  +\
                           '    ' +\
                           STR('%10.6f'%scale['s[n][1]']).rjust(10, " ")[-10:]  + STR('%10.6f'%scale['s[n][2]']).rjust(10, " ")[-10:]  + STR('%10.6f'%scale['s[n][3]']).rjust(10, " ")[-10:] +\
                           '     ' +\
                           STR('%10.5f'%scale['u[n]']).rjust(10," ")[-10:]
                fd.write(origLine)
                fd.write('\n')
        # write ter
        terLines = {}
        if ter_format:
            for key in self.ter.keys():
                ter = self.ter[key]
                terLines[key] = '%s'%STR(ter['record_name']).ljust(6, " ")[:6]  +\
                                ("%i"%ter['serial_number']).rjust(5, " ")[-5:] +\
                                '      ' +\
                                '%s'%STR(ter['residue_name']).rjust(3, " ")[:3]  +\
                                ' ' +\
                                '%s'%STR(ter['chain_identifier']).rjust(1, " ")[0] +\
                                ("%i"%ter['sequence_number']).rjust(4, " ")[-4:] +\
                                '%s'%STR(ter['code_of_insertion']).rjust(1, " ")[0]
        # write in models format
        modelsLines = {}
        endModelsLines = {}
        if model_format:
            for key in self.models.keys():
                model = self.models[key]
                modelsLines[ model['model_start'] ] = '%s'%STR(model['record_name']).ljust(6, " ")[:6]  +\
                                                      '   ' +\
                                                      ('%i'%model['model_serial_number']).rjust(5, " ")[-5:]
                termodel = model['termodel']
                endModelsLines[ model['model_end'] ] = '%s'%STR(termodel['record_name']).ljust(6, " ")[:6]  +\
                                                       ("%i"%termodel['serial_number']).rjust(5, " ")[-5:]+\
                                                       '      ' +\
                                                       '%s'%STR(termodel['residue_name']).rjust(3, " ")[:3]  +\
                                                       ' ' +\
                                                       '%s'%STR(termodel['chain_identifier']).rjust(1, " ")[0] +\
                                                       ("%i"%termodel['sequence_number']).rjust(4, " ")[-4:] +\
                                                       '%s'%STR(termodel['code_of_insertion']).rjust(1, " ")[0] +\
                                                       "\n" +\
                                                       '%s'%STR(model['endmodel']).ljust(6, " ")[:6]
        # write records
        for idx in self.indexes:
            if idx in modelsLines:
                modelLine = modelsLines[idx] + STR('\n')
            else:
                modelLine = ''
            if idx in endModelsLines:
                endModelLine = endModelsLines[idx] + STR('\n')
            else:
                endModelLine = ''
            if idx in terLines:
                terLine = terLines[idx] + STR('\n')
            else:
                terLine = ''
            # write anisou
            anisouLine = ''
            if anisou:
                 if idx in self.anisou:
                    anisou = self.anisou[idx]
                    anisouLine = '%s'%STR(anisou['record_name']).ljust(6, " ")[:6]  +\
                                 ("%i"%anisou['serial_number']).rjust(5, " ")[-5:]+\
                                 ' ' +\
                                 '%s'%STR(anisou['atom_name']).ljust(4, " ")[:4]  +\
                                 '%s'%STR(anisou['location_indicator']).rjust(1, " ")[0] +\
                                 '%s'%STR(anisou['residue_name']).rjust(3, " ")[:3]  +\
                                 ' ' +\
                                 '%s'%STR(anisou['chain_identifier']).rjust(1, " ")[0] +\
                                 ("%i"%anisou['sequence_number']).rjust(4, " ")[-4:] +\
                                 '%s'%STR(anisou['code_of_insertion']).rjust(1, " ")[0] +\
                                 ' ' +\
                                 ('%i'%anisou['u[1][1]']).rjust(7, " ")[-7:]  + ('%i'%anisou['u[2][2]']).rjust(7, " ")[-7:]  + ('%i'%anisou['u[3][3]']).rjust(7, " ")[-7:] +\
                                 ('%i'%anisou['u[1][2]']).rjust(7, " ")[-7:]  + ('%i'%anisou['u[1][3]']).rjust(7, " ")[-7:]  + ('%i'%anisou['u[2][3]']).rjust(7, " ")[-7:] +\
                                 '  ' +\
                                 '%s'%STR(anisou['segment_identifier']).ljust(4, " ")[:4] +\
                                 '%s'%STR(anisou['element_symbol']).rjust(2," ")[:2] +\
                                 '%s'%STR(anisou['charge']).rjust(2, " ")[:2]
                 else:
                    anisouLine = ''
            # write atom line
            atom = self.records[idx]
            atomLine = '%s'%STR(atom['record_name']).ljust(6, " ")[:6]  +\
                       ("%i"%atom['serial_number']).rjust(5, " ")[:5] +\
                       ' ' +\
                       '%s'%STR(atom['atom_name']).ljust(4, " ")[:4]  +\
                       '%s'%STR(atom['location_indicator']).rjust(1, " ")[0] +\
                       '%s'%STR(atom['residue_name']).rjust(3, " ")[:3] +\
                       ' ' +\
                       '%s'%STR(atom['chain_identifier']).rjust(1, " ")[0] +\
                       ("%i"%atom['sequence_number']).rjust(4, " ")[-4:] +\
                       '%s'%STR(atom['code_of_insertion']).rjust(1, " ")[0] +\
                       '   ' +\
                       ('%8.3f'%coordinates[idx,0]).rjust(8, " ")[-8:]  + ('%8.3f'%coordinates[idx,1]).rjust(8, " ")[-8:]  + ('%8.3f'%coordinates[idx,2]).rjust(8, " ")[-8:] +\
                       ('%6.2f'%atom['occupancy']).rjust(6, " ")[-6:] +\
                       ('%6.2f'%atom['temperature_factor']).rjust(6, " ")[-6:] +\
                       '      ' +\
                       '%s'%STR(atom['segment_identifier']).ljust(4, " ")[:4] +\
                       '%s'%STR(atom['element_symbol']).rjust(2, " ")[:2] +\
                       '%s'%STR(atom['charge']).rjust(2, " ")[:2]
            atomLine = atomLine.splitlines()[0] + '\n'
            fd.write(endModelLine)
            fd.write(modelLine)
            fd.write(terLine)
            fd.write(atomLine)
            fd.write(anisouLine)
        # last ENDMDL
        if len(self.records) in endModelsLines:
            fd.write( endModelsLines[ len(self.records) ] + STR('\n') )
        # last TER
        if len(self.records) in terLines:
            fd.write( terLines[ len(self.records) ] + STR('\n') )

    def set_name(self, name):
        """
        set pdbparser instance name.

        :Parameters:
            #. name (basestring): the pdb name.
        """
        if name is not None:
            assert isinstance(name, basestring), "name must be a string"
            name = STR(name)
        self.__name = name

    def set_configuration_index(self, index):
        """
        set the configuration index, for a pdbparser this method is empty.

        :Parameters:
            #. index (integer): the index of the configuration supposedly working with
        """
        return

    def set_boundary_conditions(self, bc):
        """
        set pdbparser boundary conditions instance.

        :Parameters:
            #. bc (InfiniteBoundaries, PeriodicBoundaries): the boundary conditions instance
        """
        assert isinstance(bc, (InfiniteBoundaries, PeriodicBoundaries)), "bc must be either InfiniteBoundaries or PeriodicBoundaries instance"
        self._boundaryConditions = bc
        # update crystallographicStructure
        if not isinstance(self._boundaryConditions, PeriodicBoundaries):
            a = float( 1.000 )
            b = float( 1.000 )
            c = float( 1.000 )
            alpha = float( 90.00 )
            beta  = float( 90.00 )
            gamma = float( 90.00 )
        else:
            vectors = self._boundaryConditions.get_vectors()
            angles  = self._boundaryConditions.get_angles()
            a = np.linalg.norm(vectors[0])
            b = np.linalg.norm(vectors[1])
            c = np.linalg.norm(vectors[2])
            alpha = angles[0]*180/np.pi
            beta  = angles[1]*180/np.pi
            gamma = angles[2]*180/np.pi
        self.crystallographicStructure = { "record_name": "CRYST1" ,\
                                           "a"          : a ,\
                                           "b"          : b ,\
                                           "c"          : c ,\
                                           "alpha"      : alpha ,\
                                           "beta"       : beta ,\
                                           "gamma"      : gamma ,\
                                           "space_group": "P 1       " ,\
                                           "z_value"    : 1 ,\
                                          }

    def value(self, attribute):
        """
        Get a list of all atoms attribute value.

         :Parameters:
            #. attribute (key): Any records dictionary key.
        """
        return get_records_attribute_values(self.indexes, self, attribute)

    def get_configuration_coordinates(self, index):
        """
        Get a single configuration all atoms coordinates.
        For pdbparser, it's an alias to 'coordinates' and index argument is ignored

         :Parameters:
            #. index (integer):The configuration index.
        """
        return self.coordinates

    def set_configuration_coordinates(self, index, coordinates):
        """
        set the pdb atoms coordinates. The index is ignored because pdbparser has a single configuration.

        :Parameters:
            #. index (integer): the index of the configuration.
            #. coordinates (np.array): the new atoms coordinates.
        """
        assert isinstance(coordinates, np.ndarray), Logger.error("coordinates must be numpy.ndarray instance")
        assert coordinates.shape==(self.numberOfAtoms, 3), Logger.error("coordinates shape must be %s"%((self.numberOfAtoms, 3)))
        set_coordinates(self.indexes, self, coordinates)

    def set_coordinates(self, coordinates):
        """
        alias to set_configuration_coordinates where index argument is omitted.

        :Parameters:
            #. coordinates (np.array): the new atoms coordinates.
        """
        self.set_configuration_coordinates(index=None, coordinates=coordinates)


    def set_atoms_coordinates(self, index, atomsIndexes, coordinates):
        """
        set the pdb atoms coordinates. the index is ignored because pdbparser has a single configuration.

        :Parameters:
            #. index (integer): the index of the configuration. This attribte is ignored.
            #. atomsIndexes (None, list, set, tuple): the indexes of atoms.
            #. coordinates (np.array): the new atoms coordinates.
        """
        assert isinstance(coordinates, np.ndarray), Logger.error("coordinates must be numpy.ndarray instance")
        assert coordinates.shape==(len(atomsIndexes), 3), Logger.error("coordinates shape must be %s"%((len(atomsIndexes), 3)))
        set_coordinates(atomsIndexes, self, coordinates)

    def set_records_coordinates(self, index, atomsIndexes, coordinates):
        """ alias to set_atoms_coordinates. """
        set_atoms_coordinates(index, atomsIndexes, coordinates)

    def get_records_coordinates(self, index, atomsIndexes):
        """
        get the pdb atoms coordinates.  the index is ignored because pdbparser has a single configuration.

        :Parameters:
            #. index (integer): the index of the configuration.
            #. atomsIndexes (None, list, set, tuple): the indexes of atoms.
        """
        return get_coordinates(atomsIndexes, self)

    def get_atoms_coordinates(self, index, atomsIndexes):
        """ alias to get_records_coordinates. """
        return self.get_records_coordinates(index, atomsIndexes)

    def get_contiguous_configuration_coordinates(self, configurationIndex=None, atomsIndexes=None, group=("sequence_number", "segment_identifier")):
        """
        get configuration atoms coordinates after reconstructing clusters.
        The configurationIndex argument is ignored because pdbparser has a single configuration.

        :Parameters:
            #. configurationIndex (integer): the index of the configuration.
            #. atomsIndexes (None, list, set, tuple): the indexes of atoms to return. if None, all atoms are returned.
            #. group (str, list, set, tuple): pdbparser record valid key used for grouping atoms in molecules. multiple grouping keywords is possible

        :Returns:
            #. configuration (numpy.ndarray): atom positions across all trajectory. array shape is (numberOfAtoms, 3)
        """

        if not isinstance(self._boundaryConditions, PeriodicBoundaries):
            raise Logger.error("contiguous configuration is not possible with infinite boundaries trajectory")
        # check group
        if isinstance(group, (list, set, tuple)):
            group = list(group)
            for g in group:
                if not g in self.records[0]:
                    raise Logger.error("group item %s argument must be a valid pdbparser record key"%g)
        elif not group in  self.records[0]:
            raise Logger.error("argument group %s must be a valid pdbparser record key"%group)
        else:
            group = [group]
        # get box coordinates
        if atomsIndexes is None:
            atomsIndexes = self.indexes
            coords = self.coordinates
        else:
            assert isinstance(atomsIndexes, (list, set, tuple)), Logger.error("indexes must be a list of integers")
            atomsIndexes = list(atomsIndexes)
            coords = self.coordinates[atomsIndexes]
        coords = self._boundaryConditions.real_to_box_array(realArray=coords)
        # build groups look up table
        for GRP in group:
            groups = get_records_attribute_values(atomsIndexes, self, GRP)
            groupsSet = set(groups)
            if group.index(GRP)==0:
                groupsLUT = {}
                for g in groupsSet:
                    groupsLUT[g] = []
                for idx in range(len(groups)):
                    groupsLUT[groups[idx]].append(idx)
                groups = list(groupsLUT.values())
                groupsLUT = {}
                for idx in range(len(groups)):
                    groupsLUT[idx] = groups[idx]
            else:
                for key in groupsLUT.keys():
                    indexes = groupsLUT[key]
                    grps = [groups[idx] for idx in indexes]
                    grpsSet = set(grps)
                    grsIndexes = [[indexes[idx] for idx in range(len(indexes)) if grps[idx]==g] for g in grpsSet]
                    if len(grsIndexes)>1:
                        groupsLUT[key] = grsIndexes.pop(0)
                        while grsIndexes:
                            groupsLUT[len(groupsLUT)] = grsIndexes.pop(0)
        # build contiguous configuration
        for _, groupIndexes in groupsLUT.items():
            groupCoords = coords[groupIndexes,:]
            # initialize variables
            pos = groupCoords[0,:]
            # incrementally construct cluster
            diff = groupCoords-pos
            # remove multiple box distances
            intDiff = diff.astype(int)
            groupCoords -= intDiff
            diff -= intDiff
            # remove half box distances
            groupCoords = np.where(np.abs(diff)<0.5, groupCoords, groupCoords-np.sign(diff))
            # set group atoms new box positions
            coords[groupIndexes,:] = groupCoords
        # convert box to real coordinates and return
        return self._boundaryConditions.box_to_real_array(boxArray=coords)


    def get_atom_trajectory(self, index, configurationsIndexes=None):
        """
        For pdbparser this method simply returns the coordinates of the atom

        :Parameters:
            #. index (integer): the index of the atom in structure file.
            #. configurationsIndexes (None, list): meaningless for pdbparser instance

        :Returns:
            #. trajectory (numpy.ndarray): atom positions across all trajectory. array shape is (numberOfConfigurations, 3)
        """
        rec = self.records[index]
        return np.array([rec['coordinates_x'],rec['coordinates_y'],rec['coordinates_z']])

    def remove_configuration(self, index):
        """
        remove a configuration from the trajectory. This method is simply ignored in the case of pdbparser instance.

        :Parameters:
            #. index (integer, None): the index of configuration to remove from trajectory.
        """
        pass

    def define_model(self, model_start, model_end, model_name = None):
        """
        Define a model.\n

        :Parameters:
            #. model_start (integer): the model starting index.
            #. model_end (integer): the model ending index.
            #. model_name (string): the name of the model. If None is given automatic model name generation using 'define_model_name' is used
        """
        # check if range is already allocated
        if not self.check_model_range(model_start, model_end):
            Logger.info("model cannot be defined, the given range [%s, %s] is already allocated"%(model_start, model_end))
            return
        # create model name
        if model_name is None:
            model_name = self.define_model_name()
        # create model ter
        modelRange = model_end-model_start
        ter = { "record_name"       : "TER" ,\
                "serial_number"     : modelRange+1,\
                "residue_name"      : STR( self.records[model_end-1]["residue_name"] ) ,\
                "chain_identifier"  : STR( self.records[model_end-1]["chain_identifier"] ) ,\
                "sequence_number"   : int( self.records[model_end-1]["sequence_number"] ) ,\
                "code_of_insertion" : STR( self.records[model_end-1]["code_of_insertion"] ) ,\
                "INDEX_IN_RECORDS"  : model_end ,\
              }
        # create model
        model = { "record_name"         : "MODEL" ,\
                  "model_serial_number" : len(self.models.keys())+1 ,\
                  "model_start"         : model_start ,\
                  "model_end"           : model_end ,\
                  "termodel"            : ter ,\
                  "endmodel"            : "ENDMDL",\
                  "MODEL_NAME"          : model_name
                 }
        # append model
        self.models[len(self.models.keys())+1] = model

    def define_model_name(self, basename = "model "):
        """
        Used for automatic generation of models names, an integer is automatically added to the basename insuring a unique name.\n

        :Parameters:
            #. basename (string): the basename of the model name.
        """
        modelsNames = self.get_models_names()
        modelsNum = len(self.models.keys())+1
        name = basename + STR(modelsNum)
        while name in modelsNames:
            name = name[:-1] + "0" + name[-1]
        # return name
        return name

    def check_model_range(self, model_start, model_end = -1):
        """
        Returns True if model_start AND model_end are not already in a defined model range. Otherwise Falseis returns.\n

        :Parameters:
            #. model_start (integer): the model starting index.
            #. model_end (integer): the model ending index.
        """
        for model in self.models.values():
            if (model_start<=model['model_start']<model_end) or (model_start<model['model_end']<=model_end):
                return False
        return True

    def get_models_names(self):
        """
        Returns a list of all models name.\n

        :Parameters:
            #. name (string): the model name.
        """
        return get_models_attribute_values(self, list(self.models.keys()), "MODEL_NAME")

    def get_model_key_by_name(self, name):
        """
        Returns model's key of self.models using model's name attribute.\n

        :Parameters:
            #. name (string): the model name.
        """
        return get_models_keys_by_attribute_value(list(self.models.keys()), self, "MODEL_NAME", name)

    def get_model_range_by_name(self, name):
        """
        Returns a tuple of (model_start, model_end).\n
        If name is incorrect, None value is returned.\n

        :Parameters:
            #. name (string): the model name.
        """
        return get_model_range_by_attribute_value(list(self.models.keys()), self, "MODEL_NAME", name)

    def delete_all_models_definition(self):
        """
        Deletes all models definition but not their associated records.
        """
        self.models = {}

    def read_pdb(self, filePath):
        """
        Reads and parses the pdb file and save all its records and informations. \n

        :Parameters:
            #. filePath (None, string, list): the input pdb file path or file lines. If None, pdb will be reseted
        """
        self.__reset__()
        if filePath is None:
            return
        elif isinstance(filePath, (list, tuple)):
            fd = list(filePath)
        else:
            # try to open file
            assert isinstance(filePath, basestring), Logger.error('filePath must be None, a list of lines or a string path to a pdb file')
            filePath = _normalize_path(filePath)
            try:
                fd = open(filePath, 'r')
            except:
                Logger.error("cannot open file %s" %filePath)
                raise
            else:
                self.filePath = filePath
                self.__name = os.path.basename(STR(filePath)).split('.')[0]
        # read lines
        model = None
        index = -1
        for line in fd:
            index += 1
            if line[0:6].strip() not in self.__RECORD_NAMES__:
                # headings
                if "REMARK    Boundary Conditions: " in line:
                    bcVectors = line.split("REMARK    Boundary Conditions: ")[1].strip().split()
                    if not len(bcVectors)==9:
                        Logger.warn("Wrong boundary conditions line '%s' format found. Line ignored"%line)
                        self.headings.append(line)
                    try:
                        bcVectors = [float(item) for item in bcVectors]
                        bcVectors = np.array(bcVectors)
                    except:
                        Logger.warn("Wrong boundary conditions line '%s' format found. Line ignored"%line)
                        self.headings.append(line)
                    else:
                        bc = PeriodicBoundaries()
                        bc.set_vectors(bcVectors)
                        self.set_boundary_conditions(bc)
                else:
                    self.headings.append(line)
            else:
                methodToCall = self.__RECORD_NAMES__[ line[0:6].strip() ]
                model = getattr(self, methodToCall)(line=line, model=model, index=index)
        # close file
        if not isinstance(filePath, (list, tuple)):
            fd.close()

    def export_pdb(self, outputPath ,\
                   indexes = None,\
                   additionalRemarks = None,
                   headings = True ,\
                   structure = True ,\
                   model_format = False ,\
                   ter_format = True ,\
                   origxn = True ,\
                   scalen = True ,\
                   anisou = False,\
                   coordinates=None,
                   boundaryConditions=None):
        """
        Export the current pdb into .pdb file.\n

        :Parameters:
            #. outputPath (None,string): the output pdb file path. If None
               pdb string format will be returned
            #. additionalRemarks (string): add additional Remarks to the file.
            #. headings (boolean): export including the headings.
            #. structure (boolean): export including the structure.
            #. model_format (boolean): export including models.
            #. ter_format (boolean): export including ter.
            #. origxn (boolean): export including the origxn.
            #. scalen (boolean): export including the scalen.
            #. anisou (boolean): export including the anisou.
            #. coordinates (None, np.ndarray): export pdb using different coordinates. If None, use pdb coordinates.
            #. boundaryConditions (None, PeriodicBoundaries): Export boundary conditions other than pdbparser instance ones.

        :Returns:
            #. pdb (None, str): If outputPath is not None then None is returned
               otherwise the pdb file as a string will be returned
        """
        if indexes is not None:
            pdbCopy = copy.deepcopy(self)
            pdbCopy.records = [self.records[idx] for idx in indexes]
            pdbCopy.export_pdb( outputPath=outputPath, \
                                indexes=None,\
                                additionalRemarks=additionalRemarks,\
                                headings=headings,\
                                structure=structure,\
                                model_format=model_format,\
                                ter_format=ter_format,\
                                origxn=origxn,\
                                scalen=scalen,\
                                anisou=anisou,
                                coordinates=coordinates,
                                boundaryConditions=boundaryConditions)
        else:
            # try to open file
            try:
                if outputPath is not None:
                    outputPath = _normalize_path(outputPath)
                    fd = open(outputPath, 'w')
                else:
                    from io import StringIO
                    fd = StringIO()
            except:
                raise Logger.error( "cannot open file %r for writing" %outputPath)
            # check coordinates
            if coordinates is not None:
                assert isinstance(coordinates, np.ndarray), "coordinates must be numpy array instance"
                assert coordinates.shape == (self.numberOfAtoms,3), "coordinates array shape must be (numberOfAtoms, 3)"
                assert 'float' in coordinates.dtype.name, "coordinates data type must be a float"
            else:
                coordinates = self.coordinates
            # write pdb
            self.__write_pdb(fd=fd ,\
                             headings=headings,\
                             additionalRemarks=additionalRemarks,\
                             structure=structure,\
                             model_format=model_format,\
                             ter_format=ter_format,\
                             origxn=origxn,\
                             scalen=scalen,\
                             anisou=anisou,\
                             coordinates=coordinates,
                             boundaryConditions=boundaryConditions)
            # close file
            pdbContent = None
            fileName   = outputPath
            if outputPath is None:
                pdbContent = fd.getvalue()
                fileName = "'In-memory file'"
            fd.close()
            Logger.info( "All records successfully exported to {fileName}".format(fileName=fileName))
        # return pdb lines
        return pdbContent

    def concatenate(self, pdb, boundaryConditions=None):
        """
        Concatenates given pdb instances to the current one and corrects models indexes accordingly.\n
        If given pdb has no defined models, this is the same as: self.records.extend(pdb.records).\n

        :Parameters:
            #. pdb (pdbparser): The pdb instance to concatenate.
            #. boundaryConditions (None, pdbparser.boundaryConditions, list): The new boundary conditions. list of number boundary conditions vectors convertible must be given. if None boundary conditions will get reset to infinite boundaries.
        """
        assert isinstance(pdb, pdbparser), Logger.error( "pdb must be a pdbparser instance")
        # reset boundary conditions to infinite
        if boundaryConditions is None:
            self._boundaryConditions = InfiniteBoundaries()
        elif isinstance(boundaryConditions, (InfiniteBoundaries, PeriodicBoundaries)):
            self._boundaryConditions = boundaryConditions
        else:
            try:
                bc=PeriodicBoundaries()
                bc.set_vectors(boundaryConditions)
            except:
                raise Logger.error( "boundary conditions argument can't be converted to pdbparser BoundaryConditions instance")
            else:
                self._boundaryConditions = bc
        recordsLength = len(self.records)
        # extend records
        self.records.extend(pdb.records)
        # change pdb models indexes
        lenModels = len(self.models.keys())
        for key in pdb.models.keys():
            self.models[key+lenModels] = copy.deepcopy(pdb.models[key])
            self.models[key+lenModels] ["model_start"] += recordsLength
            self.models[key+lenModels] ["model_end"] += recordsLength
            self.models[key+lenModels] ["termodel"]["INDEX_IN_RECORDS"] += recordsLength
        # change pdb ter indexes
        for key in pdb.ter.keys():
            newKey = key+recordsLength
            self.ter[key+recordsLength] = copy.deepcopy(pdb.ter[key])
            ter["INDEX_IN_RECORDS"] = newKey


    def get_copy(self, indexes = None):
        """
        Get a copy of the current pdb, correcting the models indexes.
        If current pdb has no defined models and indexes is set to None, this is the same as: copy.deepcopy(self)\n
        If a model has records not among indexes, model definition is merely deleted.

        :Parameters:
            #. indexes (list, tuple, None): The pdb records indexes to copy. None returns all indexes.

        :Returns:
            #. pdbCopy (pdbparser): The copied pdb instance.
        """
        if indexes == None:
            indexes = self.indexes
        # get deep copy
        pdbCopy = copy.deepcopy(self)
        if not len(self.models):
            pdbCopy.records = [pdbCopy.records[idx] for idx in indexes]
        else:
            # invert indexes
            indexes = [idx for idx in self.indexes if idx not in indexes]
            delete_records(indexes, pdbCopy)
        # return pdb
        return pdbCopy

    def visualize(self, *args, **kwargs):
        """alias method. by default this alias points to visualize_vmd"""
        self.visualize_vmd(*args, **kwargs)
        
    def visualize_vmd(self, commands=None, indexes = None, coordinates = None, vmdAlias = None, startupScript=None):
        """
        Visualize current pdb using VMD software.\n

        :Parameters:
            #. commands (None, list, tuple): List of commands to pass upon calling vmd.
               commands can be a .dcd file to load a trajectory for instance.
            #. indexes (None, list, tuple): The atoms indexes to visualize. if None, all atoms are visualized.
            #. coordinates (None, np.ndarray): Change atoms coordinates for visualization purposes. If None records coordinates will be used.
            #. vmdAlias (string): VMD executable path. If None is given, VMD alias of pdbparser parameter file is used.
            #. startupScript (string): The startup file path. Launch vmd and pass .tcl script.
        """
        # create tempfile
        (fd, filename) = tempfile.mkstemp(suffix='.pdb')
        # export temporary file
        self.export_pdb(filename, indexes, headings = False ,\
                                  additionalRemarks = None,\
                                  structure = True ,\
                                  model_format = False ,\
                                  ter_format = True ,\
                                  origxn = True ,\
                                  scalen = True ,\
                                  anisou = False,
                                  coordinates=coordinates)
        # visualize
        if vmdAlias is None:
            vmdAlias = get_vmd_path()
        if vmdAlias is None:
            Logger.warn("vmd software path is not found. pdbparser.Globals.PREFERENCES.update_preferences({'VMD_PATH':'path to vmd executable'}) ")
            return
        if commands is None:
            commands = ""
        else:
            assert isinstance(commands, (list,tuple)), Logger.error("commands must be a list")
            for c in commands:
                assert isinstance(c, basestring) , Logger.error("each command must be a string")
            commands = " ".join(commands)
        try:
            if startupScript is not None:
                os.system("%s -pdb %s %s -e %s" %(vmdAlias, filename, commands, startupScript) )
            else:
                os.system("%s -pdb %s %s" %(vmdAlias, commands, filename) )
        except:
            os.remove(filename)
            raise Logger.error('vmd alias %r defined is not correct' %vmdAlias)
        else:
            # remove tempfile
            os.remove(filename)


    def visualize_pymol(self, indexes=None, coordinates=None, pymolAlias=None):
        """
        Visualize current pdb using pymol software.\n

        :Parameters:
            #. indexes (None, list, tuple): The atoms indexes to visualize. if None, all atoms are visualized.
            #. coordinates (None, np.ndarray): Change atoms coordinates for visualization purposes. If None records coordinates will be used.
            #. pymolAlias (string): pymol executable path. If None is given, pymol alias of pdbparser parameter file is used.
            #. startupScript (string): The startup file path. Launch vmd and pass .tcl script.
        """

        # create tempfile
        (fd, filename) = tempfile.mkstemp(suffix='.pdb')
        # export temporary file
        self.export_pdb(filename, indexes, headings = False ,\
                                  additionalRemarks = None,\
                                  structure = True ,\
                                  model_format = False ,\
                                  ter_format = True ,\
                                  origxn = True ,\
                                  scalen = True ,\
                                  anisou = False,
                                  coordinates=coordinates)
        if pymolAlias is None:
            pymolAlias = get_pymol_path()
        if pymolAlias is None:
            Logger.warn("pymol software path is not found. pdbparser.Globals.PREFERENCES.update_preferences({'PYMOL_PATH':'path to pymol executable'}) ")
            return
        try:
            os.system("%s %s" %(pymolAlias, filename) )
        except:
            os.remove(filename)
            raise Logger.error('pymol alias %r defined is not correct' %vmdAlias)
        else:
            # remove tempfile
            os.remove(filename)


class pdbTrajectory(object):
    def __init__(self, trajectory=None):
        """
        pdb trajectory class. Coordinates are in Angstrom and time is in ps

        :Parameters:
            #. trajectory (string): unless set to None, trajectory file path to load.
        """
        self.__reset__()
        # load trajectory
        if trajectory is not None:
            self.load(trajectory)

    def __len__(self):
        return len(self._coordinates)

    def __reset__(self):
        self._filePath = None
        self.__configurationIndex = 0
        self._structure = pdbparser()
        self._coordinates = []
        self._time = []
        self._boundaryConditions = None
        self._info = {}

    @property
    def info(self):
        """
        returns the informations stored in the trajectory
        """
        return self._info

    @property
    def filePath(self):
        """
        returns the loaded trajectory file path
        """
        return self._filePath

    @property
    def numberOfConfigurations(self):
        """
        returns the number of configurations. it is the same as len(pdbTrajectory)
        """
        return len(self)

    @property
    def configurationIndex(self):
        """
        returns the configuration index supposedly working with
        """
        return self.__configurationIndex

    @property
    def atomsIndexes(self):
        """
        returns a list of all structure atoms indexes.
        """
        return self._structure.indexes

    @property
    def indexes(self):
        """
        returns a list of all structure atoms indexes.
        """
        return range(len(self))

    @property
    def xindexes(self):
        """
        returns a list of all structure atoms indexes.
        """
        return xrange( len(self) )

    @property
    def xatomsIndexes(self):
        """
        returns a generator of all structure atoms indexes.
        """
        return self._structure.xindexes

    @property
    def numberOfAtoms(self):
        """
        returns number of atoms.
        """
        return len(self._structure)

    @property
    def time(self):
        """
        returns the trajectory time as numpy.array copy.
        """
        return np.array(self._time)

    @property
    def structure(self):
        """
        return a copy of structure pdbparser instance.
        """
        return copy.deepcopy(self._structure)

    @property
    def coordinates(self):
        """
        return all configuration coordinates at index=configurationIndex
        """
        return self._coordinates[self.__configurationIndex]

    @property
    def coordinatesArray(self):
        """
        return all coordinates as numpy.array copy of shape (numberOfConfigurations,numberOfAtoms,3)
        """
        return np.array(self._coordinates, dtype=np.float32)

    @property
    def simulationBox(self):
        """
        return the simulation box instance
        """
        return self._boundaryConditions

    @property
    def boundaryConditions(self):
        """
        returns the bounday condition instance.
        """
        return self._boundaryConditions

    @property
    def elements(self):
        """
        returns a list of all structure elements
        """
        return self._structure.elements

    @property
    def names(self):
        """
        returns a list of all structure atoms name
        """
        return self._structure.names

    @property
    def residues(self):
        """
        returns a list of all structure residue name
        """
        return self._structure.residues

    @property
    def sequences(self):
        """
        returns a list of all structure sequence number
        """
        return self._structure.sequences

    @property
    def segments(self):
        """
        returns a list of all structure segment identifier
        """
        return self._structure.segments

    @property
    def records(self):
        """
        returns a list of all structure records
        """
        return self._structure.records

    def set_configuration_index(self, index):
        """
        set the configuration index

        :Parameters:
            #. index (integer): the index of the configuration supposedly working with
        """
        try:
            index = int(index)
        except:
            raise Logger.error("index must be integer convertible")
        assert index>=0 and index<len(self), Logger.error("index must within trajectory length")
        self.__configurationIndex = index

    def set_structure(self, structure):
        """
        Set a new structure file. All trajectory attribute will be reset

        :Parameters:
            #. structure (string, pdbparser): the new structure pdbparser instance or pdb file path.
        """
        if not isinstance(structure, pdbparser):
            try:
                structure = pdbparser(structure)
            except:
                raise Logger.error("pdbTrajectory structure must be pdbparser convertible file or None")
        self.__reset__()
        self._structure = structure

    def save(self, path):
        """
        save pdbTrajectory in a binary file

        :Parameters:
            #. path (string): file path
        """
        assert isinstance(path, basestring), Logger.error("path must be a string")
        path = STR(path).split(".pdbt")[0]+".pdbt"
        pickle.dump( self, open( path, "wb" ), protocol=2 )

    def load(self, path):
        """
        load pdbTrajectory binary file

        :Parameters:
            #. path (string): file path
        """
        assert isinstance(path, basestring), Logger.error("path must be a string")
        try:
            traj = pickle.load( open( path, "rb" ) )
        except:
            raise Logger.error("cannot open %s"%path)
        else:
            assert isinstance(traj, pdbTrajectory), Logger.error("object cannot be converted to pdbTrajectory")
        # set filePath
        self._filePath = path
        # set structure
        self._structure = traj._structure
        # set time array
        self._time  = traj._time
        # set coordinates
        self._coordinates = traj._coordinates
        # set boundary conditions
        try:
            self._boundaryConditions = traj._boundaryConditions
        except:
            self._boundaryConditions = traj._simulationBox
        # set trajectory info
        try:
            self._info = traj._info
        except:
            self._info = {}


    def visualize(self, indexes=None, atomsIndexes=None, vmdAlias = None):
        """
        set the configuration index

        :Parameters:
            #. indexes (list): the configuration indexes to visualize. If None all configurations are loaded
            #. atomsIndexes (list): the atoms indexes to visualize. If None all atoms are loaded
            #. vmdAlias (string): VMD executable path. If None is given, VMD alias of pdbparser parameter file is used.
        """
        # create pdbTrajectory temporary folder
        try:
            tempDir = tempfile.mkdtemp()
        except:
            raise Logger.error("cannot create temporary directory")
        # create temporary trajectory file
        filename = os.path.join(tempDir, "trajectory.xyz")
        self.export_xyz_trajectory(outputPath=filename, indexes=indexes, atomsIndexes=atomsIndexes)
        # visualize
        if vmdAlias is None:
            vmdAlias = get_vmd_path()
        if vmdAlias is None:
            Logger.warn("vmd software path is not found. pdbparser.Globals.PREFERENCES.update_preferences({'VMD_PATH':'path to vmd executable'}) ")
            return
        try:
            os.system( "%s %s" %(vmdAlias, filename) )
        except:
            shutil.rmtree(tempDir)
            raise Logger.error('vmd alias %r defined is not correct' %vmdAlias)
        # remove trajectory file
        shutil.rmtree(tempDir)

    def convert_NAMD_trajectory(self, pdb, xst, dcd, indexes=None, vmdAlias=None):
        """
        Read new simulation trajectory

        :Parameters:
            #. pdb (string): NAMD pdb file used as trajectory structure file
            #. xst (string): NAMD xst output file
            #. dcd (string): NAMD DCD output file
            #. indexes (list): the configuration indexes to convert. None converts all configurations
            #. vmdAlias (string): VMD executable path. If None is given, VMD alias of pdbparser parameter file is used.
        """
        Logger.info("Converting NAMD trajectory")
        Logger.info("pdb file path: %s"%pdb)
        Logger.info("xst file path: %s"%xst)
        Logger.info("dcd file path: %s"%dcd)
        # check indexes
        if indexes is not None:
            assert isinstance(indexes, (list, tuple, set, np.ndarray)), Logger.error("indexes must be a list of 3 integers [start, end, step]")
            indexes = list(indexes)
            assert len(indexes)==3 , Logger.error("indexes must be a list of 3 integers [start, end, step]")
            try:
                indexes = [int(indexes[0]), int(indexes[1]), int(indexes[2])]
            except:
                raise Logger.error("indexes must be a list of 3 integers [start, end, step]")
            assert indexes[0]>=1, Logger.error("indexes start must be positive")
            assert indexes[1]>=1, Logger.error("indexes end must be positive")
            assert indexes[2]>=1, Logger.error("indexes step must be positive")
            assert indexes[0]<indexes[1], Logger.error("indexes start must be smalled then end")
        # check pdb file
        try:
            fd = open(pdb,'r')
        except:
            raise Logger.error("cannot open pdb file")
        else:
            fd.close()
        # check xst file
        try:
            fd = open(xst,'r')
        except:
            raise Logger.error("cannot open xst file")
        else:
            fd.close()
        # check dcd file
        try:
            fd = open(dcd,'r')
        except:
            raise Logger.error("cannot open dcd file")
        else:
            fd.close()
        # create pdbTrajectory temporary folder
        try:
            tempDir = tempfile.mkdtemp()
        except:
            raise Logger.error("cannot create temporary directory")
        # create vmd log file
        vmdLog = os.path.join(tempDir,"vmd.log")
        # read xst file and remove first line which is not savec in dcd file
        xstData = np.loadtxt(xst)
        times   = xstData[1:,0]/1000.
        vectors = xstData[1:,1:10]
        xstNumConfig = len(times)
        if indexes is None:
            indexes = [0, len(times),1]
        else:
            times = times[range(*indexes),:]
            vectors = vectors[range(*indexes),:]
        # create vmd extract pdb files
        extractFiles_file = os.path.join(tempDir,"extractFiles.png")
        try:
            fd = open(extractFiles_file, 'w')
        except:
            shutil.rmtree(tempDir)
            raise Logger.error("cannot create temporary files")
        else:
            fd.write("mol load pdb %s dcd %s \n"%(pdb,dcd))
            fd.write("set nf [molinfo top get numframes] \n")
            fd.write('if {$nf != %s} then { \n' %(xstNumConfig+1))
            fd.write(        'puts "dcd number of frames $nf while xst is %s" \n'%(len(times)+1) )
            fd.write(        "exit }\n")
            fd.write("for { set i %s } {$i < %s } { set i [expr $i + %s]} { \n"%(indexes[0],indexes[1],indexes[2]))
            fd.write("        set sel [atomselect top all frame $i] \n")
            fd.write("        $sel writepdb %s \n" %os.path.join(tempDir,"$i.pdb") )
            fd.write("}\n")
            fd.write("exit")
            fd.close()

        # export trajectory files
        if vmdAlias is None:
            vmdAlias = get_vmd_path()
        if vmdAlias is None:
            Logger.warn("vmd software path is not found. pdbparser.Globals.PREFERENCES.update_preferences({'VMD_PATH':'path to vmd executable'}) ")
            return
        try:
            Logger.info("exporting trajectory files")
            os.system( "%s -dispdev text -e  %s > %s" %(vmdAlias, extractFiles_file, vmdLog) )
        except:
            shutil.rmtree(tempDir)
            raise Logger.error('vmd alias %r defined is not correct' %vmdAlias)
        else:
            os.remove(extractFiles_file)
        # get number of pdb created
        numberOfFrames = len([name for name in os.listdir(tempDir) if ".pdb" in name])
        if len(times)!=numberOfFrames:
            shutil.rmtree(tempDir)
            raise Logger.error("xst number of frames is %s must be equal to exported dcd number of frames which is %s"%(len(times),numberOfFrames))
        # set structure
        try:
            self.set_structure(pdb)
        except:
            shutil.rmtree(tempDir)
            raise Logger.error("cannot read pdb structure file")
        # read files
        pdbFilesIndexes = range(indexes[0],indexes[1],indexes[2])
        for idx in range(numberOfFrames):
            Logger.info('reading configuration index %i' %pdbFilesIndexes[idx])
            try:
                pdbPath = os.path.join(tempDir,"%s.pdb"%pdbFilesIndexes[idx])
                self.append_configuration(pdbPath, vectors=vectors[idx], time=times[idx])
                os.remove(pdbPath)
            except Exception as e:
                shutil.rmtree(tempDir)
                raise Logger.error(e)
        # remove temp directory
        shutil.rmtree(tempDir)

    def concatenate(self, trajectory, adjustTime=True, highLevel=False):
        """
        Concatenate with trajectory. trajectories must have same number of atoms

        :Parameters:
            #. trajectory (string, pdbparser.pdbTrajectory): the trajectory path or instance to concatenate.
            #. correctTime (boolean): adjust trajectory time to ensure final concatenated trajectory time continuity.
            #. highLevel (boolean): compatibility level. if not low, records names are checked between current structure and new trajectory structure
        """
        assert trajectory is not self, Logger.error("self-concatenation is dangerous. Use a copy.")
        assert isinstance(highLevel, bool), Logger.error("highLevel must be boolean")
        if not isinstance(trajectory, pdbTrajectory):
            try:
                trajectory = pickle.load( open( trajectory, "rb" ) )
            except:
                raise Logger.error("cannot open %s"%trajectory)
            else:
                assert isinstance(trajectory, pdbTrajectory), Logger.error("trajectory must be pdbTrajectory convertible")
        # check periodic boundary conditions
        assert self._boundaryConditions.__class__.__name__==trajectory._simulationBox.__class__.__name__, Logger.error("boundaries conditions doesn't match")
        # check number of atoms
        if self.numberOfAtoms and len(self):
            assert self.numberOfAtoms==trajectory.numberOfAtoms, Logger.error("trajectory and existing structure must have the same number of atoms")
        # check complete compatibility
        if highLevel:
            assert compare_two_lists(self._structure.names, trajectory._structure.names), Logger.error("records names doesn't match")
            assert compare_two_lists(self._structure.elements, trajectory._structure.elements), Logger.error("records elements doesn't match")
            assert compare_two_lists(self._structure.residues, trajectory._structure.residues), Logger.error("records residues doesn't match")
        # concatenate
        timeCorrection = 0
        if adjustTime:
            timeCorrection += self._time[-1]-trajectory._time[0]
            if len(trajectory)>1:
                timeCorrection += trajectory._time[1]-trajectory._time[0]
            elif len(self)>1:
                timeCorrection += self._time[1]-self._time[0]
            elif self._times[0]!=0:
                timeCorrection += self._time[0]
            else:
                raise Logger.error("time adjustement is not possible")
        self._coordinates.extend(trajectory._coordinates)
        for idx in range(len(trajectory)):
            self._boundaryConditions.set_vectors(trajectory._simulationBox.get_vectors(idx))
            self._time.append(timeCorrection+trajectory._time[idx])

    def remove_configuration(self, index):
        """
        remove a configuration from the trajectory.

        :Parameters:
            #. index (integer, None): the index of configuration to remove from trajectory.
        """
        # check index
        try:
            index = float(index)
        except:
            raise Logger.error("configuration index must be number convertible, %s given instead"%index)
        assert index>=0, Logger.error("configuration index must be positive number")
        assert index%1==0, Logger.error("configuration index must be number and integer equivalent")
        assert index<=len(self), Logger.error("configuraion index must not exceed trajectory length %s"%len(self))
        index = int(index)
        # remove configuration
        self._boundaryConditions.delete_vectors(index)
        self._time.pop(index)
        self._coordinates.pop(index)

    def remove_configurations(self, indexes):
        """
        remove a configuration from the trajectory.

        :Parameters:
            #. indexes (integer, list, set, tuple): the list of atoms indexes in structure file.
        """
        if not isinstance(indexes, (list, set, tuple)):
            indexes = [indexes]
        else:
            indexes = list(indexes)
        for index in indexes:
            try:
                index = float(index)
            except:
                raise Logger.error("configuration index must be number convertible, %s given instead"%index)
            assert index>=0, Logger.error("configuration index must be positive number")
            assert index%1==0, Logger.error("configuration index must be number and integer equivalent")
            assert index<len(self), Logger.error("atom index must smaller than number of atoms in structure %s"%len(self._structure))
        # set indexes in order and remove redundancy
        indexes = sorted(set(indexes))
        # remove configs
        removed=0
        for idx in indexes:
            index = int(idx-removed)
            # remove configuration
            self._boundaryConditions.delete_vectors(index)
            self._time.pop(index)
            self._coordinates.pop(index)
            #increment remove
            removed += 1

    def insert_configuration(self, index, pdb, vectors, time):
        """
        insert configuration coordinates to the trajectory at a given index.

        :Parameters:
            #. index (integer, None): the index of insertion in trajectory.
            #. pdb (path, pdbparser): the configuration pdb. Number of atoms in pdb must be equal to the structure one.
            #. vectors (None, number, numpy.ndarray):  The simulation box vectors. None value designate infinite simulation box. Consistency is verified, if first configuration is periodic all appended ones must be periodic as well, same apply for infinite simulation box.
            #. time (number): The time of the configuration. Must be bigger then the last configuration time.
        """
        # check structure file
        assert isinstance(self._structure, pdbparser), "must define a structure file before adding configurations"
        # check index
        try:
            index = float(index)
        except:
            raise Logger.error("configuration index must be number convertible, %s given instead"%index)
        assert index>=0, Logger.error("configuration index must be positive number")
        assert index%1==0, Logger.error("configuration index must be number and integer equivalent")
        assert index<=len(self), Logger.error("configuraion index must not exceed trajectory length %s"%len(self))
        index = int(index)
        # check time
        try:
            time = float(time)
        except:
            raise Logger.error("configuration time must be number convertible, %s given instead"%time)
        assert time>=0, Logger.error("configuration time must be positive number")
        if index == 0:
            if len(self._time) > 0:
                assert time<self._time[0], Logger.error("time must be incremental, given time for first configuration is %s while current first given configuration time is %s"%(time,self._time[0]))
        elif index == len(self._time):
            assert time>self._time[-1], Logger.error("time must be incremental, given time for last configuration is %s while current last configuration is %s"%(time,self._time[-1]))
        else:
            assert time>self._time[index-1], Logger.error("time must be incremental, given time for index %s is %s while existing time at configuration (index-1=%s) is %s"%(index,time,index-1, self._time[index-1]))
            assert time<self._time[index], Logger.error("time must be incremental, given time for index %s is %s while existing time at configuration (index=%s) is %s"%(index, time,index, self._time[index]))
        # check pdb
        if not isinstance(pdb, pdbparser):
            try:
                pdb = pdbparser(pdb)
            except:
                raise Logger.error("pdb must be pdbparser convertible file")
        assert len(pdb)==len(self._structure), Logger.error("number of atoms in pdb doesn't match witht the number of atoms in trajectory structure")
        # check vectors
        assert (vectors is None) or (isinstance(vectors, (Number, np.ndarray))), Logger.error("vectors can be either None, number, numpy.ndarray")
        if self._boundaryConditions is None:
            if vectors is None:
                sb = InfiniteBoundaries()
            else:
                sb = PeriodicBoundaries()
                try:
                    sb.set_vectors(vectors)
                except:
                    raise Logger.error("incompatible vectors format")
        ############ Done with checking ###########
        # add simulation box vectors
        if self._boundaryConditions is None:
            self._boundaryConditions = sb
        else:
            if isinstance(self._boundaryConditions, PeriodicBoundaries):
                assert vectors is not None, Logger.error("simulation box is registered as periodic, None vectors values are not accepted")
            else:
                assert vectors is None, Logger.error("simulation box is registered as infinite, only None vectors values are accepted")
            try:
                self._boundaryConditions.set_vectors(vectors, index)
            except:
                raise Logger.error("incompatible vectors format")
        # add time
        self._time.insert(index, time)
        # add configuration coordinates
        self._coordinates.insert(index, np.float32(pdb.coordinates))

    def append_configuration(self, pdb, vectors, time):
        """
        append configuration coordinates to the trajectory.

        :Parameters:
            #. pdb (path, pdbparser): the configuration pdb. Number of atoms in pdb must be equal to the structure one.
            #. vectors (None, number, numpy.ndarray):  The simulation box vectors. None value designate infinite simulation box. Consistency is verified, if first configuration is periodic all appended ones must be periodic as well, same apply for infinite simulation box.
            #. time (number): The time of the configuration. Must be bigger then the last configuration time.
        """
        self.insert_configuration(index=len(self), pdb=pdb, vectors=vectors, time=time)

    def get_configuration_coordinates(self, index):
        """
        get configuration atoms coordinates as saved in trajectory without boundary conditions corrections.

        :Parameters:
            #. index (integer): the index of the configuration.

        :Returns:
            #. configuration (numpy.ndarray): atom positions across all trajectory. array shape is (numberOfAtoms, 3)
        """
        try:
            index = float(index)
        except:
            raise Logger.error("configuration index must be number convertible, %s given instead"%index)
        assert index>=0, Logger.error("configuration index must be positive number")
        assert index%1==0, Logger.error("configuration index must be number and integer equivalent")
        assert index<len(self), Logger.error("configuraion index must not smaller then trajectory length %s"%len(self))
        index = int(index)
        return self._coordinates[index]

    def get_contiguous_configuration_coordinates(self, configurationIndex, atomsIndexes=None, group=("sequence_number", "segment_identifier")):
        """
        get configuration atoms coordinates after reconstructing clusters.

        :Parameters:
            #. configurationIndex (integer): the index of the configuration.
            #. group (str, list, set, tuple): pdbparser record valid key used for grouping atoms in molecules. multiple grouping keywords is possible
            #. atomsIndexes (None, list, set, tuple): the indexes of atoms to return. if None, all atoms are returned.

        :Returns:
            #. configuration (numpy.ndarray): atom positions across all trajectory. array shape is (numberOfAtoms, 3)
        """
        if not isinstance(self._boundaryConditions, PeriodicBoundaries):
            raise Logger.error("contiguous configuration is not possible with infinite boundaries trajectory")
        try:
            configurationIndex = float(configurationIndex)
        except:
            raise Logger.error("configuration index must be number convertible, %s given instead"%index)
        assert configurationIndex>=0, Logger.error("configuration index must be positive number")
        assert configurationIndex%1==0, Logger.error("configuration index must be number and integer equivalent")
        assert configurationIndex<len(self), Logger.error("configuraion index must not smaller then trajectory length %s"%len(self))
        configurationIndex = int(configurationIndex)
        # check group
        if isinstance(group, (list, set, tuple)):
            group = list(group)
            for g in group:
                if not g in self._structure.records[0]:
                    raise Logger.error("group item %s argument must be a valid pdbparser record key"%g)
        elif not group in self._structure.records[0]:
            raise Logger.error("argument group %s must be a valid pdbparser record key"%group)
        else:
            group = [group]
        # get box coordinates
        if atomsIndexes is None:
            atomsIndexes = self._structure.indexes
            coords = self._coordinates[configurationIndex]
        else:
            assert isinstance(atomsIndexes, (list, set, tuple)), Logger.error("indexes must be a list of integers")
            atomsIndexes = list(atomsIndexes)
            coords = self._coordinates[configurationIndex][atomsIndexes]
        coords = self._boundaryConditions.real_to_box_array(realArray=coords, index=configurationIndex)
        # build groups look up table
        for GRP in group:
            groups = get_records_attribute_values(atomsIndexes, self._structure, GRP)
            groupsSet = set(groups)
            if group.index(GRP)==0:
                groupsLUT = {}
                for g in groupsSet:
                    groupsLUT[g] = []
                for idx in range(len(groups)):
                    groupsLUT[groups[idx]].append(idx)
                groups = list(groupsLUT.values())
                groupsLUT = {}
                for idx in range(len(groups)):
                    groupsLUT[idx] = groups[idx]
            else:
                for key in groupsLUT.keys():
                    indexes = groupsLUT[key]
                    grps = [groups[idx] for idx in indexes]
                    grpsSet = set(grps)
                    grsIndexes = [[indexes[idx] for idx in range(len(indexes)) if grps[idx]==g] for g in grpsSet]
                    if len(grsIndexes)>1:
                        groupsLUT[key] = grsIndexes.pop(0)
                        while grsIndexes:
                            groupsLUT[len(groupsLUT)] = grsIndexes.pop(0)
        # build contiguous configuration
        for _, groupIndexes in groupsLUT.items():
            groupCoords = coords[groupIndexes,:]
            # initialize variables
            pos = groupCoords[0,:]
            # incrementally construct cluster
            diff = groupCoords-pos
            # remove multiple box distances
            intDiff = diff.astype(int)
            groupCoords -= intDiff
            diff -= intDiff
            # remove half box distances
            groupCoords = np.where(np.abs(diff)<0.5, groupCoords, groupCoords-np.sign(diff))
            # set group atoms new box positions
            coords[groupIndexes,:] = groupCoords
        # convert box to real coordinates and return
        return self._boundaryConditions.box_to_real_array(boxArray=coords, index=configurationIndex)

    def set_configuration_coordinates(self, index, coordinates):
        """
        set configuration atoms coordinates.

        :Parameters:
            #. index (integer): the index of the configuration.
            #. coordinates (np.array): the new atoms coordinates.
        """
        try:
            index = float(index)
        except:
            raise Logger.error("configuration index must be number convertible, %s given instead"%index)
        assert index>=0, Logger.error("configuration index must be positive number")
        assert index%1==0, Logger.error("configuration index must be number and integer equivalent")
        assert index<len(self), Logger.error("configuraion index must not smaller then trajectory length %s"%len(self))
        index = int(index)
        assert isinstance(coordinates, np.ndarray), Logger.error("coordinates must be numpy.ndarray instance")
        assert coordinates.shape==(self.numberOfAtoms, 3), Logger.error("coordinates shape must be %s"%((self.numberOfAtoms, 3)))
        self._coordinates[index] =  coordinates

    def set_atoms_coordinates(self, index, atomsIndexes, coordinates):
        """
        set the pdb atoms coordinates.

        :Parameters:
            #. index (integer): the index of the configuration.
            #. atomsIndexes (None, list, set, tuple): the indexes of atoms.
            #. coordinates (np.array): the new atoms coordinates.
        """
        try:
            index = float(index)
        except:
            raise Logger.error("configuration index must be number convertible, %s given instead"%index)
        assert index>=0, Logger.error("configuration index must be positive number")
        assert index%1==0, Logger.error("configuration index must be number and integer equivalent")
        assert index<len(self), Logger.error("configuraion index must not smaller then trajectory length %s"%len(self))
        index = int(index)
        assert isinstance(coordinates, np.ndarray), Logger.error("coordinates must be numpy.ndarray instance")
        # create atomsIndexes list
        for idx in atomsIndexes:
            try:
                idx = float(idx)
            except:
                raise Logger.error("atom idx must be number convertible, %s given instead"%index)
            assert idx>=0, Logger.error("atom idx must be positive number")
            assert idx%1==0, Logger.error("atom idx must be number and integer equivalent")
            assert idx<len(self._structure), Logger.error("atom idx must smaller than number of atoms in structure %s"%len(self._structure))
        atomsIndexes = [int(idx) for idx in atomsIndexes]
        # check coordinates shape
        assert coordinates.shape==(len(atomsIndexes), 3), Logger.error("coordinates shape must be %s"%((self.numberOfAtoms, 3)))
        # set coordinates
        self._coordinates[index][atomsIndexes:] =  coordinates

    def set_atoms_coordinates(self, index, atomsIndexes, coordinates):
        """
        alias to set_atoms_coordinates
        """
        self.set_atoms_coordinates(index, atomsIndexes, coordinates)

    def remove_atoms(self, indexes):
        """
        remove atom from trajectory.

        :Parameters:
            #. indexes (integer, list, set, tuple): the list of atoms indexes in structure file.
        """
        if not isinstance(indexes, (list, set, tuple)):
            indexes = [indexes]
        else:
            indexes = list(indexes)
        for index in indexes:
            try:
                index = float(index)
            except:
                raise Logger.error("atom index must be number convertible, %s given instead"%index)
            assert index>=0, Logger.error("atom index must be positive number")
            assert index%1==0, Logger.error("atom index must be number and integer equivalent")
            assert index<len(self._structure), Logger.error("atom index must smaller than number of atoms in structure %s"%len(self._structure))
        # create indexes list
        indexes = [int(idx) for idx in indexes]
        # atoms indexes to keep
        atomsIndexesToKeep = sorted(set(self.atomsIndexes)-set(indexes))
        # remove atom from structure
        records = self._structure.records
        self._structure.records = [records[idx] for idx in atomsIndexesToKeep]
        # remove atom from trajectory
        for confIdx in range(len(self._coordinates)):
            self._coordinates[confIdx] = self._coordinates[confIdx][atomsIndexesToKeep,:]


    def get_atom_coordinates(self, index, configurationsIndexes=None):
        """
        get atom position as saved in trajectory without boundary conditions corrections.

        :Parameters:
            #. index (integer): the index of the atom in structure file.
            #. configurationsIndexes (None, list): list of frames indexes. if None all frames are considered.

        :Returns:
            #. positions (numpy.ndarray): atom positions across all trajectory. array shape is (numberOfConfigurations, 3)
        """
        try:
            index = float(index)
        except:
            raise Logger.error("atom index must be number convertible, %s given instead"%index)
        assert index>=0, Logger.error("atom index must be positive number")
        assert index%1==0, Logger.error("atom index must be number and integer equivalent")
        assert index<len(self._structure), Logger.error("configuraion index must smaller than number of atoms in structure %s"%len(self._structure))
        index = int(index)
        if configurationsIndexes is None:
            configurationsIndexes = self.xindexes
        return np.array([self._coordinates[fidx][index,:] for fidx in configurationsIndexes])

    def get_atom_trajectory(self, index, configurationsIndexes=None):
        """
        get atom position real trajectory with boundary conditions corrections. when trajectory output frequency is poor and atoms may jump longer than half of simulation box between successive configuration, the output trajectory becomes not possible to calculate and the output may be erroneous.

        :Parameters:
            #. index (integer): the index of the atom in structure file.
            #. configurationsIndexes (None, list): list of frames indexes. if None all frames are considered.

        :Returns:
            #. trajectory (numpy.ndarray): atom positions across all trajectory. array shape is (numberOfConfigurations, 3)
        """
        # get all position for better correction
        positions = self.get_atom_coordinates(index)
        # one configuration trajectory
        if positions.shape[0] == 1:
            if configurationsIndexes is None:
                configurationsIndexes = self.xindexes
            return np.array([positions[idx,:] for idx in configurationsIndexes])

        # periodic bounday conditions
        if isinstance(self._boundaryConditions, PeriodicBoundaries):
            # correct positions
            boxPositions = np.array([self._boundaryConditions.real_to_box_array(positions[idx], idx) for idx in self.xindexes])
            boxDifferences = boxPositions[1:,:]-boxPositions[:-1,:]
            foldedDifference = np.where(np.abs(boxDifferences)<0.5, 0, boxDifferences+np.sign(boxDifferences)*(1-abs(boxDifferences)))
            boxPositions[1:,:] -= np.cumsum(foldedDifference, axis=0)
            #if np.any(np.abs(boxDifferences)>0.5) and index>1000:
            #    import matplotlib.pylab as plt
            #    plt.plot(self.time, np.array([self._boundaryConditions.real_to_box_array(positions[idx], idx) for idx in self.xindexes]), label="before")
            #    plt.plot(self.time, boxPositions, label="after")
            #    plt.title(STR(index))
            #    plt.legend()
            #    plt.show()
            # return
            if configurationsIndexes is None:
                return np.array([self._boundaryConditions.box_to_real_array(boxPositions[idx], idx) for idx in self.xindexes])
            else:
                return np.array([self._boundaryConditions.box_to_real_array(boxPositions[idx], idx) for idx in configurationsIndexes])
        # infinite boundaries
        else:
            if configurationsIndexes is None:
                return np.array([positions[idx] for idx in self.xindexes])
            else:
                return np.array([positions[idx] for idx in configurationsIndexes])


    def get_configuration_pdb(self, configurationIndex):
        """
        get pdbparser instance of a single configuration.

        :Parameters:
            #. index (integer): the index of the atom in structure file.
            #. configurationIndex (integer): configuration index.

        :Returns:
            #. pdb (pdbparser.pdbparser): pdbparser instance at configurationIndex
        """
        coords = self.get_configuration_coordinates(configurationIndex)
        pdbCopy = copy.deepcopy(self._structure)
        set_coordinates(pdbCopy.xindexes, pdbCopy, coords)
        return pdbCopy

    def export_xyz_trajectory(self, outputPath, indexes=None, atomsIndexes=None):
        """
        Export the current pdb into .pdb file.\n

        :Parameters:
            #. outputPath (string): the output pdb file path and name. configuration index will be added automatically to the file name.
            #. indexes (None, list): the configuration indexes to visualize. If None all configurations are loaded
            #. atomsIndexes (None, list): the atoms indexes to visualize. If None all atoms are loaded
        """
        # check indexes
        if indexes is None:
            indexes = self.indexes
        else:
            assert isinstance(indexes, (list, set, tuple)), Logger.error("indexes must be a list of positive integers smaller than trajectory length")
            indexes = sorted(set(indexes))
            assert not len([False for idx in indexes if (idx%1!=0 or idx<0 or idx>=len(self))]), Logger.error("indexes must be a list of positive integers smaller than trajectory length")
            indexes = [int(idx) for idx in set(indexes)]
        # check indexes
        if atomsIndexes is None:
            atomsIndexes = self.atomsIndexes
        else:
            assert isinstance(atomsIndexes, (list, set, tuple)), Logger.error("atomsIndexes must be a list of positive integers")
            atomsIndexes = sorted(set(atomsIndexes))
            assert not len([False for idx in atomsIndexes if (idx%1!=0 or idx<0 or idx>=self.numberOfAtoms)]), Logger.error("atomsIndexes must be a list of positive integers smaller than the total number of atoms")
            atomsIndexes = [int(idx) for idx in set(atomsIndexes)]
        # check outputPath
        assert isinstance(outputPath, basestring), "outputPath must be a string"
        assert not os.path.isfile(outputPath), "'%s' already exist"%outputPath
        # open file
        fd = open(outputPath, 'a')
        # write trajectory file
        atomNames = [STR(name).strip().rjust(5) for name in self.names]
        for frameIndex in indexes:
            fd.write("%s\n"%len(atomsIndexes))
            fd.write("pdbTrajectory. configuration %s,  time %s \n"%(frameIndex, self._time[frameIndex]) )
            coords = self._coordinates[frameIndex]
            frame = [atomNames[idx]+ " " + "%10.5f"%coords[idx][0] + " %10.5f"%coords[idx][1] + " %10.5f"%coords[idx][2] + "\n" for idx in atomsIndexes]
            fd.write("".join(frame))
        # close file
        fd.close()

    def export_pdbs(self, outputPath = None,\
                          basename = "",\
                          configurationsIndexes = None,\
                          atomsIndexes = None):
        """
        Export trajectory configurations into seperate pdb files.\n

        :Parameters:
            #. outputPath (string): the output pdb file path and name. configuration index will be added automatically to the file name.
            #. basename (string): pdb files basename.
            #. configurationsIndexes (None, list, set, tuple): the indexes of configurations. if None, all configurations are exported.
            #. atomsIndexes (None, list, set, tuple): the indexes of atoms. if None, all atoms are exported.
        """
        # check outputPath
        assert isinstance(outputPath, str), Logger.error("outputPath must be a string valid path")
        assert isinstance(basename, str), Logger.error("basename must be a string")
        # check configurationsIndexes
        if configurationsIndexes is None:
            configurationsIndexes = self.indexes
        else:
            assert isinstance(configurationsIndexes, (list, set, tuple)), Logger.error("configurationsIndexes must be a list of positive integers")
            configurationsIndexes = list(configurationsIndexes)
            for index in configurationsIndexes:
                try:
                    index = float(index)
                except:
                    raise Logger.error("configuration index must be number convertible, %s given instead"%index)
                assert index>=0, Logger.error("configuration index must be positive number")
                assert index%1==0, Logger.error("configuration index must be number and integer equivalent")
                assert index<len(self), Logger.error("configuraion index must smaller than number of configurations in trajectory %s"%len(self))
            configurationsIndexes = sorted(set([int(float(index)) for index in configurationsIndexes]))
        # check atomsIndexes
        if atomsIndexes is None:
            atomsIndexes = self._structure.indexes
        else:
            assert isinstance(atomsIndexes, (list, set, tuple)), Logger.error("atomsIndexes must be a list of positive integers")
            atomsIndexes = list(atomsIndexes)
            for index in atomsIndexes:
                assert is_number(index), Logger.error("atom index must be number convertible, %s given instead"%index)
                index = int(float(index))
                assert index>=0, Logger.error("atom index must be positive number")
                assert index%1==0, Logger.error("atom index must be number and integer equivalent")
                assert index<len(self), Logger.error("atom index must smaller than number of atoms in structure %s"%len(self._structure))
            atomsIndexes = sorted(set([int(float(index)) for index in atomsIndexes]))
        # get copy of the structure
        pdbCopy = copy.deepcopy(self._structure)
        pdbCopy.records = [pdbCopy.records[idx] for idx in atomsIndexes]
        # get trajectory number of digits in number of configurations
        ndigits = int(np.log10(len(self))+1)
        # start export loop
        for idx in configurationsIndexes:
            path = os.path.join(outputPath,basename)+"_"+STR(idx).rjust(ndigits,'0')+".pdb"
            # set coordinates
            set_coordinates(pdbCopy.xindexes, pdbCopy, self._coordinates[idx][atomsIndexes,:])
            # create header
            header = []
            header.append( "pdbTrajectory file path:                            %s\n"%STR(self._filePath) )
            for key, val in self._info.items():
                header.append(STR("simulation info - %s: "%(STR(key))).ljust(52)+ STR(val) + "\n")
            header.append( "trajectory units for time and positions:            picosecond and angstrom\n")
            header.append( "trajectory number of configurations at export time: %s\n"%STR(len(self)) )
            header.append( "configuration index exported in this file:          %s\n"%STR(idx) )
            header.append( "configuration time exported in this file:           %s ps\n"%STR(self._time[idx]) )
            header.append( "trajectory total number of atoms:                   %s\n"%STR(len(self._structure)) )
            header.append( "number of atoms exported in this pdb:               %s\n"%STR(len(atomsIndexes)) )
            header.append( "simulation boundary conditions:                     %s\n"%STR(self._boundaryConditions.__class__.__name__))
            if isinstance(self._boundaryConditions, PeriodicBoundaries):
                vectors = self._boundaryConditions.get_vectors(idx)
                header.append( "a = %.6f     %.6f     %.6f\n"%tuple(vectors[0,:]) )
                header.append( "b = %.6f     %.6f     %.6f\n"%tuple(vectors[1,:]) )
                header.append( "c = %.6f     %.6f     %.6f\n"%tuple(vectors[2,:]) )
            # output
            pdbCopy.export_pdb( path,\
                                additionalRemarks = header ,\
                                headings = False ,\
                                structure = False ,\
                                model_format = False ,\
                                ter_format = False ,\
                                origxn = False ,\
                                scalen = False ,\
                                anisou = False)
