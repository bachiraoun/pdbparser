"""
In this test, an SDS molecule is loaded
several records manipulations, translation, rotation, orientation, ... are tested
"""
# standard distribution imports
from __future__ import print_function
import os

# pdbparser imports
from pdbparser.Utilities.Collection import get_path
from pdbparser.log import Logger
from pdbparser.pdbparser import pdbparser
from pdbparser.Utilities.Geometry import *

pdbRESULT = pdbparser()

Logger.info("loading sds molecule ...")
pdbSDS = pdbparser(os.path.join(get_path("pdbparser"),"Data","SDS.pdb" ) )
INDEXES = range(len(pdbSDS.records))
# get molecule axis
sdsAxis = get_axis(INDEXES,  pdbSDS)
 ## translate to positive quadrant
atomToOriginIndex = get_closest_to_origin(INDEXES, pdbSDS)
atom = pdbSDS.records[atomToOriginIndex]
[minX, minY, minZ]  = [ atom['coordinates_x'] , atom['coordinates_y'] , atom['coordinates_z'] ]
translate(INDEXES, pdbSDS, [-1.1*minX, -1.1*minY, -1.1*minZ])


Logger.info("orient molecule along [1,0,0] ...")
orient(axis=[1,0,0], indexes=INDEXES, pdb=pdbSDS, records_axis=sdsAxis)
sdsAxis = [1,0,0]
pdbRESULT.concatenate(pdbSDS)

Logger.info("Flip molecule 180 degrees ...")
pdb = pdbSDS.get_copy()
orient(axis=[-1,1,0], indexes=INDEXES, pdb=pdb, records_axis=sdsAxis)
pdbRESULT.concatenate(pdb)

pdbRESULT.visualize()
