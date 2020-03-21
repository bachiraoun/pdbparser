"""
This module provides classes used in the Input-Output processes.

.. inheritance-diagram:: pdbparser.IO.Core
    :parts: 2

"""
# standard libraries imports
from __future__ import print_function
from copy import copy
import os
import re
import struct

# external libraries imports
import numpy as np

# pdbparser library imports
from pdbparser.log import Logger


class FortranBinaryFile(object):
    """
    Sets up a Fortran binary file reader.
    """
    def __init__(self, filename, byte_order = '='):
        """
        The constructor.

        :Parameters:
            #. filename (string): the binary input file
            #. byte_order (string): the byte order to read the binary file. Must be any of '@', '=', '<', '>' or '!'.
        """
        self.__file = file(filename, 'rb')
        self.__byteOrder = byte_order
        self.__fileSize = os.path.getsize(filename)

    def __iter__(self):
        return self

    @property
    def fileSize(self):
        return self.__fileSize

    @property
    def currentPosition(self):
        return self.__file.tell()

    def next(self):
        data = self.__file.read(4)
        if not data:
            raise StopIteration
        reclen = struct.unpack(self.__byteOrder + 'i', data)[0]
        data = self.__file.read(reclen)
        reclen2 = struct.unpack(self.__byteOrder + 'i', self.__file.read(4))[0]
        assert reclen==reclen2, Logger.error("data format not respected")
        return data

    def skip_record(self):
        data = self.__file.read(4)
        reclen = struct.unpack(self.__byteOrder + 'i', data)[0]
        self.__file.seek(reclen, 1)
        reclen2 = struct.unpack(self.__byteOrder + 'i', self.__file.read(4))[0]
        assert reclen==reclen2, Logger.error("data format not respected")

    def get_record(self, format, repeat = False):
        """
        Reads a record of the binary file.

        :Parameters:
            #. format (string): the format corresponding to the binray structure to read.
            #. repeat (boolean): if True, will repeat the reading.
        """
        try:
            data = self.next()
        except StopIteration:
            raise Logger.error("Unexpected end of file")
        if repeat:
            unit = struct.calcsize(self.__byteOrder + format)
            assert len(data) % unit == 0, Logger.error("wrong data length")
            format = (len(data)/unit) * format
        try:
            return struct.unpack(self.__byteOrder + format, data)
        except:
            raise Logger.error("not able to unpack data")


class DCDFile(object):
    """
    sets up a DCD file reader.
    """

    def __init__(self, filename):
        """
        The constructor.

        :Parameters:
            #. filename (string): the binary input file
        """
        # time unit in charmm
        self.charmmTimeToPs = 0.0488882129084
        # Identity the byte order of the file by trial-and-error
        self.__byteOrder = None
        data = file(filename, 'rb').read(4)
        for byte_order in ['<', '>']:
            reclen = struct.unpack(byte_order + 'i', data)[0]
            if reclen == 84:
                self.__byteOrder = byte_order
                break
        if self.__byteOrder is None:
            raise Logger.error("%s is not a DCD file" % filename)
        # Open the file
        self.__binary = FortranBinaryFile(filename, self.__byteOrder)
        # Read the header information
        header_data = self.__binary.next()
        if header_data[:4] != 'CORD':
            raise Logger.error("%s is not a DCD file" % filename)
        self.header = struct.unpack(self.__byteOrder + '9id9i', header_data[4:])
        self.numberOfConfigurations = self.header[0]
        self.istart = self.header[1]
        self.nsavc = self.header[2]
        self.namnf = self.header[8]
        self.charmmVersion = self.header[-1]
        self.has_pbc_data = False
        self.has_4d = False
        if self.charmmVersion != 0:
            self.header = struct.unpack(self.__byteOrder + '9if10i',
                                        header_data[4:])
            if self.header[10] != 0:
                self.has_pbc_data = True
            if self.header[11] != 0:
                self.has_4d = True
        self.delta = self.header[9]*self.charmmTimeToPs
        # Read the title
        title_data = self.__binary.next()
        nlines = struct.unpack(self.__byteOrder + 'i', title_data[:4])[0]
        assert len(title_data) == 80*nlines+4, Logger.error("%s is not a DCD file" % filename)
        title_data = title_data[4:]
        title = []
        for i in range(nlines):
            title.append(title_data[:80].rstrip())
            title_data = title_data[80:]
        self.title = '\n'.join(title)
        # Read the number of atoms.
        self.natoms = self.__binary.get_record('i')[0]
        # Stop if there are fixed atoms.
        if self.namnf > 0:
            raise Logger.error("NAMD converter can not handle fixed atoms yet")

    @property
    def fileSize(self):
        return self.__binary.fileSize

    @property
    def currentPosition(self):
        return self.__binary.currentPosition

    def read_step(self):
        """
        Reads a configuration of the DCD file.
        """
        if self.has_pbc_data:
            unit_cell = np.array(self.__binary.get_record('6d'), dtype = np.float)
            a, gamma, b, beta, alpha, c = unit_cell
            if -1. < alpha < 1. and -1. < beta < 1. and -1. < gamma < 1.:
                # assume the angles are stored as cosines
                # (CHARMM, NAMD > 2.5)
                alpha = 0.5*np.pi-np.arcsin(alpha)
                beta = 0.5*np.pi-np.arcsin(beta)
                gamma = 0.5*np.pi-np.arcsin(gamma)
            unit_cell = (a, b, c, alpha, beta, gamma)
        else:
            unit_cell = None
        format = '%df' % self.natoms
        x = np.array(self.__binary.get_record(format), dtype = np.float32)
        y = np.array(self.__binary.get_record(format), dtype = np.float32)
        z = np.array(self.__binary.get_record(format), dtype = np.float32)
        if self.has_4d:
            self.__binary.skip_record()
        return unit_cell, x, y, z

    def skip_step(self):
        """
        Skips a configuration of the DCD file.
        """
        nrecords = 3
        if self.has_pbc_data:
            nrecords += 1
        if self.has_4d:
            nrecords += 1
        for i in range(nrecords):
            self.__binary.skip_record()

    def __iter__(self):
        return self

    def next(self):
        try:
            return self.readStep()
        except:
            raise StopIteration



class Converter(object):
    def __init__(self):
        self.__previousStep = None
        pass

    def status(self, step, totalSteps, logFrequency = 10):
        """
        This method is used to log converting status.\n

        :Parameters:
            #. step (int): The current step number
            #. logFrequency (float): the frequency of status logging. its a percent number.
        """
        if not step:
            Logger.info("%s --> converting started" %(self.__class__.__name__))
        elif step == totalSteps:
            Logger.info("%s --> converting finished" %(self.__class__.__name__))
        else:
            actualPercent = int( float(step)/float(totalSteps)*100)
            if self.__previousStep is not None:
                previousPercent = int(float(self.__previousStep)/float(totalSteps)*100)
            else:
                previousPercent = -1
            if actualPercent/logFrequency != previousPercent/logFrequency:
                Logger.info("%s --> %s%% completed. %s left out of %s" %(self.__class__.__name__, actualPercent, totalSteps-step, totalSteps))
        # update previous step
        self.__previousStep = step
