from __future__ import print_function
import sys, os
from pypref import Preferences as PREF


class Preferences(PREF):
    __thisInstance = None
    def __new__(cls, *args, **kwargs):
        if cls.__thisInstance is None:
            cls.__thisInstance = super(Preferences,cls).__new__(cls)
            cls.__thisInstance._isInitialized = False
        return cls.__thisInstance

    def __init__(self, *args, **kwargs):
        if (self._isInitialized): return
        super(Preferences, self).__init__(*args, **kwargs)
        self._isInitialized = True

    def custom_init(self):
        prefs = {}
        # find vmd path if it exist
        exePath = self.preferences.get("VMD_PATH", "")
        if not os.path.exists(exePath):
            if sys.platform == "win32":
                exePath = None
                for p in [fname for fname in os.listdir("C:\\") if "Program Files" in fname]:
                    path = os.path.join("C:\\", p, "University of Illinois", "VMD", "vmd.exe")
                    if os.path.exists(path):
                        exePath = path
                prefs["VMD_PATH"] = exePath
            elif sys.platform == "darwin":
                exePath = None
                if os.path.exists('/Applications'):
                    exePath = '/Applications'
                    files = [fname for fname in os.listdir(exePath) if 'VMD' in fname]
                    if not len(files):
                        exePath = None
                if exePath is not None:
                    exePath = os.path.join(exePath, files[0], 'Contents', 'vmd')
                    files = [fname for fname in os.listdir(exePath) if 'vmd_MACOS' in fname]
                    if not len(files):
                        exePath = None
                if exePath is not None:
                    exePath = os.path.join(exePath, files[0])
                prefs["VMD_PATH"] = exePath
            else:
                exePath = None
                if os.path.exists("/usr/local/bin/vmd"):
                    exePath = "/usr/local/bin/vmd"
        if exePath is None:
            exePath = ''
        prefs["VMD_PATH"] = exePath

        ## find pymol path if exists
        exePath = self.preferences.get("PYMOL_PATH", "")
        if not os.path.exists(exePath):
            if sys.platform == "win32":
                exePath = "C:\\Program/ Files\\PyMOL\\PyMOL"
                if not os.path.exist(exePath):
                    exePath = None
            elif sys.platform == "darwin":
                #"/Applications/PyMOL.app/Contents/bin/pymol"
                exePath = None
                if os.path.exists("/Applications"):
                    exePath = "/Applications"
                    files = [fname for fname in os.listdir(exePath) if 'PyMOL.app' in fname]
                    if not len(files):
                        exePath = None
                if exePath is not None:
                    exePath = os.path.join(exePath, files[0], 'Contents', 'bin', 'pymol')
                    if not os.path.exist(exePath):
                        exePath = None
            else:
                exePath = "/usr/local/bin/pymol"
                if not os.path.exists("/usr/local/bin/pymol"):
                    exePath = None
        if exePath is None:
            exePath = ''
        prefs["PYMOL_PATH"] = exePath
        ## check if preference are set
        p = self.preferences
        if not all([prefs[k]==p[k] if k in p else False for k in prefs]):
            p.update(prefs)
            self.set_preferences(prefs)

PREFERENCES = Preferences(filename="pdbparserParams.py")
