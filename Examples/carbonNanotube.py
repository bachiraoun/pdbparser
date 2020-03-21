"""
Construct a graphene sheet in two orientations
"""
from __future__ import print_function
from pdbparser.log import Logger
from pdbparser import pdbparser
from pdbparser.Utilities.Geometry import translate
from pdbparser.Utilities.Construct import Sheet, Nanotube, MultipleWallNanotube

Logger.info("Constructing arm-chair sheet")
pdbGS_armchair = Sheet().construct().get_pdb()

Logger.info("Constructing zig-zag sheet")
pdbGS_zigzag = Sheet(orientation = "zigzag").construct().pdb
translate(pdbGS_zigzag.indexes, pdbGS_zigzag, vector = [0,0,10])

Logger.info("Constructing carbon nanotube from scratch")
pdbCNT1 = Sheet(orientation = "zigzag").construct().wrap().get_pdb()
translate(pdbCNT1.indexes, pdbCNT1, vector = [0,40,30])

Logger.info("constructing nanotube using appropriate class")
pdbCNT2 = Nanotube().construct().get_pdb()
translate(pdbCNT2.indexes, pdbCNT2, vector = [0,10,30])

Logger.info("constructing 5 walls multi-walled nanotube")
pdbMWNT = MultipleWallNanotube(wallsNumber = 5,
                               orientation = ["armchair", "zig-zag","zig-zag","zig-zag","armchair"]).construct().get_pdb()
translate(pdbMWNT.indexes, pdbMWNT, vector = [0,25,-40])

pdbALL = pdbGS_armchair
pdbALL.concatenate(pdbGS_zigzag)
pdbALL.concatenate(pdbCNT1)
pdbALL.concatenate(pdbCNT2)
pdbALL.concatenate(pdbMWNT)

pdbALL.visualize()
