"""
This modules provides methods to perform symmetry operations on a pdbparser instance atoms.

.. inheritance-diagram:: pdbparser.Utilities.Symmetry
    :parts: 2
"""
# standard libraries imports
from __future__ import print_function

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser.Utilities.Geometry import *
from pdbparser.Utilities.Information import *
from pdbparser.Utilities.Collection import get_orthonormal_axes


def inverte(indexes, pdb, vector, inversionCenter = None):
    if inversionCenter is None:
        multiply(indexes, pdb, [-1,-1,-1])
    else:
        # translate to inversionCenter
        translate(indexes, pdb, -1*np.array(inversionCenter))
        multiply(indexes, pdb, [-1,-1,-1])
        # translate back to [0,0,0]
        translate(indexes, pdb, np.array(inversionCenter))


def mirror(indexes, pdb, plane = None, origin = None):
    if plane is None:
        plane = [[1,0,0],[0,1,0]]
    else:
        assert isinstance(plane, (list,tuple))
        plane = list(plane)
        assert len(plane) == 2
        plane = [np.array(item) for item in plane]
        assert plane[0].shape in ((3,),(3,1))
        assert plane[1].shape in ((3,),(3,1))

    if origin is None:
        origin = np.array([0,0,0])
    else:
        assert isinstance(origin, (list,tuple, np.ndarray))
        origin = np.array(origin)
        assert origin.shape in ((3,),(3,1))
    # get plane axes matrix
    axes = get_orthonormal_axes(plane[0],plane[1], force = False)
    axes = np.array(axes)
    inversedAxes = np.linalg.inv(axes)
    # translate to origin
    translate(indexes, pdb, -1.0*origin)
    # change coordinates reference to axes
    rotate(indexes, pdb, inversedAxes)
    # invert Z in new reference
    multiply(indexes, pdb, [1,1,-1])
    # change to original reference
    rotate(indexes, pdb, axes)
    # return pdb
    return pdb
