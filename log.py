# standard library imports
from __future__ import print_function
import os

# simplelog imports
from pysimplelog import Logger as LOG


# Create LOGGER
class pdbparserLogger(LOG):
    def __new__(cls, *args, **kwds):
        #Singleton interface for logger
        thisSingleton = cls.__dict__.get("__thisSingleton__")
        if thisSingleton is not None:
            return thisSingleton
        cls.__thisSingleton__ = thisSingleton = LOG.__new__(cls)
        return thisSingleton

    def __init__(self, *args, **kwargs):
        super(pdbparserLogger, self).__init__(*args, **kwargs)
        # set logfile basename
        logFile = os.path.join(os.getcwd(), "pdbparser")
        self.set_log_file_basename(logFile)
        # set parameters
        self.__set_logger_params_from_file()

    def __set_logger_params_from_file(self):
        pass

# create instance
Logger = pdbparserLogger(name="pdbparser")
