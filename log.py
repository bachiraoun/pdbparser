# standard library imports
from __future__ import print_function
import os

# simplelog imports
from pysimplelog import Logger as LOG


## OLD IMPLEMENTATION — kept for reference
# class pdbparserLogger(LOG):
#     def __new__(cls, *args, **kwds):
#         #Singleton interface for logger
#         thisSingleton = cls.__dict__.get("__thisSingleton__")
#         if thisSingleton is not None:
#             return thisSingleton
#         cls.__thisSingleton__ = thisSingleton = LOG.__new__(cls)
#         return thisSingleton
#
#     def __init__(self, *args, **kwargs):
#         super(pdbparserLogger, self).__init__(*args, **kwargs)
#         # set logfile basename
#         logFile = os.path.join(os.getcwd(), "pdbparser")
#         self.set_log_file_basename(logFile)
#         # set parameters
#         self.__set_logger_params_from_file()
#
#     def __set_logger_params_from_file(self):
#         pass
#
# # create instance
# Logger = pdbparserLogger(name="pdbparser")


class _Logger(object):
    """
    Singleton holder for a pysimplelog.Logger instance configured for pdbparser.

    All attribute access and assignment delegate transparently to self.logger
    via __getattr__ / __setattr__, so every call site in pdbparser works with
    no changes.  The inner logger can be replaced at runtime via set_logger().
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, name):
        # guard against re-initialization on repeated _Logger() calls
        if 'logger' in object.__getattribute__(self, '__dict__'):
            return
        object.__setattr__(self, 'logger', LOG(name=name))
        self.logger.set_log_file_basename(os.path.join(os.getcwd(), "pdbparser"))

    def set_logger(self, loggerInstance):
        """
        Swap the inner logger instance.

        :Parameters:
            #. loggerInstance (LOG): replacement logger — must be a
               pysimplelog.Logger instance (or subclass such as SingleLogger).
        """
        assert isinstance(loggerInstance, LOG), "loggerInstance must be a pysimplelog.Logger instance, got %s" % type(loggerInstance)
        object.__setattr__(self, 'logger', loggerInstance)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'logger'), name)

    def __setattr__(self, name, value):
        if name == 'logger':
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, 'logger'), name, value)

    def __repr__(self):
        return repr(object.__getattribute__(self, 'logger'))


# initialize Logger
Logger = _Logger(name="pdbparser")
