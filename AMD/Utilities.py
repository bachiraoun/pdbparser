#!/Library/Frameworks/Python.framework/Versions/7.0/Resources/Python.app/Contents/MacOS/Python
import numpy as np

def randomVelocity(temperature, mass, kb=1.3806513e-23):
    """
    creates a random velocity array of (N,3) shape
    from the given (N,1) mass at the given temperature
    mass in kilogram and temperature in Kelvin
    """
    #kB = 1.3806513e-23 J/K
    sigma = np.divide( np.sqrt(temperature*kb), mass)
    return np.concatenate( [np.reshape( np.random.normal(0, sigma, (mass.shape) ), (-1,1) ),\
                            np.reshape( np.random.normal(0, sigma, (mass.shape) ), (-1,1) ),\
                            np.reshape( np.random.normal(0, sigma, (mass.shape) ), (-1,1) )], axis = 1 )



def readFileCheck(filePath):
    try:
      open(filePath, "r")
      return True
    except IOError:
      return False



class cprint(object):
    """
    cprint is a class that prints colored texts
    it can be imported and used without initialization.
    """

    # colors code
    CODE = {}
    CODE["off"] = '\033[0m'

    CODE["bold"] = '\033[1m'
    CODE["underline"] = '\033[4m'
    CODE["blinking"] = '\033[5m'
    CODE["highlight"] = '\033[7m'
    CODE["hide"] = '\033[8m'
    CODE["strike"] = '\033[9m'

    CODE["black"] = '\033[30m'
    CODE["red"] = '\033[31m'
    CODE["green"] = '\033[32m'
    CODE["yellow"] = '\033[33m'
    CODE["blue"] = '\033[34m'
    CODE["magenta"] = '\033[35m'
    CODE["cyan"] = '\033[36m'
    CODE["grey"] = '\033[37m'

    CODE["black_background"] = '\033[40m'+CODE["grey"]
    CODE["red_background"] = '\033[41m'
    CODE["green_background"] = '\033[42m'
    CODE["yellow_background"] = '\033[43m'
    CODE["blue_background"] = '\033[44m'+CODE["grey"]
    CODE["magenta_background"] = '\033[45m'
    CODE["cyan_background"] = '\033[46m'
    CODE["grey_background"] = '\033[47m'

    CODE["title"] = CODE["red"]+CODE["bold"]+CODE["underline"]
    CODE["subtitle"] = CODE["bold"]+CODE["underline"]
    CODE["header"] = CODE["underline"]
    CODE["warning"] = CODE["blue"]
    CODE["information"] = CODE["green"]
    CODE["error"] = CODE["red"]
    CODE["fail"] = CODE["bold"]


    # create singleton
    _instance = False
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance


    def __init__(self,*args):
        self.__call__(*args)


    def __call__(self, statement, color_code = 'black', end_color = True):
        """
        statement is the string to print
        color_code is the color to set, it could be a string as well as a list of strings
        end_color if set to False all the following printing will be colored, even normal python print
        """
        # get color code
        try:
            if isinstance(color_code, list):
                colorCode = "".join( [self.CODE[color.lower()] for color in color_code] )
            else:
                colorCode = self.CODE[color_code.lower()]
        except:
             colorCode = color_code

        # print
        if end_color:
            print colorCode + statement + self.CODE["off"]
        else:
            print colorCode + statement



class StructureFileParser(object):
    """
    structure file parser. parses atoms, mass, charges, bonds, dihedrals ...
    """
    def __init__(self, structure_file, atoms = True):
        if not readFileCheck(structure_file):
            raise "unable to open structure file"

        self.structureFile = structure_file

        # atoms section
        self.numberOfAtoms = None
        self.residueNumber = None
        self.residueName = None
        self.atomName = None
        self.forceFieldType = None
        self.atomCharge = None
        self.atomMass = None

        lines = self.readLines()
        if atoms:
            self.parseAtoms(lines)


    def readLines(self):
        fd = open(self.structureFile, 'r')
        lines = fd.readlines()
        fd.close()
        return lines


    def parseAtoms(self, lines = None):
        """
        parses the atoms section of the structure file
        """
        idxStart = None
        for idx in range(len(lines)):
            line = lines[idx]
            if "!NATOM" in line:
                idxStart = idx+1
                self.numberOfAtoms = float(line.split()[0])
                self.residueNumber = np.empty((self.numberOfAtoms), dtype=int)
                self.residueName = np.empty((self.numberOfAtoms),dtype = '|S4')
                self.atomName = np.empty((self.numberOfAtoms),dtype = '|S4')
                self.forceFieldType = np.empty((self.numberOfAtoms),dtype = '|S4')
                self.atomCharge = np.empty((self.numberOfAtoms), dtype=float)
                self.atomMass = np.empty((self.numberOfAtoms), dtype=float)
                break

        if idxStart is None:
            raise "!NATOM field not found in structure file"

        for idx in range(idxStart,len(lines)):
            arrayIdx = int(idx - idxStart)
            if arrayIdx == self.numberOfAtoms:
                break

            line = lines[idx].split()

            if len(line) == 0 and arrayIdx<self.numberOfAtoms:
                raise "number of atoms not respected in structure file"

            if arrayIdx+1 != int(line[0]):
                raise "atoms not correct in structure file"

            # 0-atomNumber 1-segment 2-residueNumber 3-resiudeName 4-atomName 5-fftype 6-charge 7-mass
            self.residueNumber[arrayIdx] = int(line[2])
            self.residueName[arrayIdx] = str(line[3])
            self.atomName[arrayIdx] =  str(line[4])
            self.forceFieldType[arrayIdx] = str(line[5])
            self.atomCharge[arrayIdx] = float(line[6])
            self.atomMass[arrayIdx] = float(line[7])



class Array(np.ndarray):
    """
    This is a classical numpy array with fast exclude and include slicing options for axis 0
    """
    def __new__(cls, value, *args, **kwargs):
        obj = np.asarray(value).view(cls)
        obj.__extras = list(kwargs.keys())
        for k,v in kwargs.items():
            setattr(obj,k,v)
        return obj


    def __init__(self, array):
        self.revealed = np.array( [True]*self.shape[0] )


    def __array_finalize__(self, obj):
        if obj is None:
            return
        for att in getattr(obj,'__extras',[]):
            setattr(self,att,getattr(obj,att))


    def reveal(self, indexes):
        """
        unmask indexes
        """
        self.revealed[indexes] = True


    def mask(self, indexes):
        """
        mask indexes
        """
        self.revealed[indexes] = False


    def mask_all(self):
        """
        mask all
        """
        self.revealed[:] = False


    def reveal_all(self):
        """
        mask all
        """
        self.revealed[:] = True


    def copy_slice(self):
        """
        Return copy of selected slices of an array along axis 0.
        slices are adjusted using reveal and mask methods
        """
        return np.compress(condition = self.revealed, a = self, axis = 0, out = None)



if __name__ == "__main__":
    # color_print
    colors = ["bold", "underline", "highlight", "blinking","strike",\
              "title","subtitle","header", "error", "fail", "warning", "information"\
              "black", "blue", "green", "red", "yellow", "magenta", "cyan", "grey",\
              "black_background", "red_background" ,"green_background", "yellow_background", "blue_background", "magenta_background", "cyan_background", "grey_background",\
              ["Red", "Bold", "Underline" ], ["Blue", "Blinking", "Highlight" ], ["Yellow", "Green", "Blue", "Red"] ]

    for color in colors:
        cprint("".join(color), color )




#     # StructureFileParser
#     print "\n"
#     sfp = StructureFileParser("/Users/AOUN/Desktop/Collaboration/micelles/ctab_c16/solvatedMicelle.psf")
#     print sfp.atomName
#     print sfp.residueNumber
#     print sfp.residueName
#     print sfp.atomName
#     print sfp.forceFieldType
#     print sfp.atomCharge
#     print sfp.atomMass
#     print sfp.numberOfAtoms
