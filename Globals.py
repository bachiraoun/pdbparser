from __future__ import print_function
import sys, os
from pypref import SinglePreferences

class Preferences(SinglePreferences):
    def custom_init(self):
        prefs = {}
        # find vmd path if it exist
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
            prefs["VMD_PATH"] = exePath
        ## check if preference are set
        if sorted(self.preferences.keys()) != sorted(prefs.keys()):
            self.set_preferences(prefs)

PREFERENCES = Preferences(filename="pdbparserParams.py")
