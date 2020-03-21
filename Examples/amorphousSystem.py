# standard distribution imports
import os

# pdbparser imports
from pdbparser.Utilities.Collection import get_path
from pdbparser import pdbparser
from pdbparser.Utilities.Construct import AmorphousSystem
from pdbparser.Utilities.Database import __WATER__

# create pdbWATER
pdbWATER = pdbparser()
pdbWATER.records = __WATER__
pdbWATER.set_name("water")

# get pdb molecules
pdbDMPC = pdbparser(os.path.join(get_path("pdbparser"),"Data","DMPC.pdb" ) )
pdbNAGMA = pdbparser(os.path.join(get_path("pdbparser"),"Data","NAGMA.pdb" ) )
pdbNALMA = pdbparser(os.path.join(get_path("pdbparser"),"Data","NALMA.pdb" ) )

# construct amorphous system, adding restrictions and existing micelle in universe
pdbAMORPH = AmorphousSystem([pdbWATER, pdbDMPC, pdbNAGMA, pdbNALMA],
                             boxSize = [150,150,150],
                             density = 0.25,
                             restrictions = "np.sqrt(x**2+y**2+z**2)<25" ).construct()

# visualize amorphous system
pdbAMORPH.get_pdb().visualize()
