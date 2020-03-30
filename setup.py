"""
In order to work properly, this script must be put one layer/folder/directory
outside of pdbparser package directory.
"""
try:
    from setuptools import setup
except:
    from distutils.core import setup
import fnmatch
from distutils.util import convert_path
import os, sys, glob

# set package path and name
PACKAGE_PATH = '.'
PACKAGE_NAME = 'pdbparser'

# check python version
major, minor = sys.version_info[:2]
if major==2 and minor!=7:
    raise RuntimeError("Python version 2.7.x or >=3.x is required.")

# automatically create MANIFEST.in
commands = [# include MANIFEST.in
            '# include this file, to ensure we can recreate source distributions',
            'include MANIFEST.in'
            # exclude all .log files
            '\n# exclude all logs',
            'global-exclude *.log',
            # exclude all pdbparserParams files
            '\n# exclude all pdbparserParams files',
            'global-exclude *pdbparserParams.*',
            # exclude all other non necessary files
            '\n# exclude all other non necessary files ',
            'global-exclude .project',
            'global-exclude .pydevproject',
            # exclude all of the subversion metadata
            '\n# exclude all of the subversion metadata',
            'global-exclude *.svn*',
            'global-exclude .svn/*',
            'global-exclude *.git*',
            'global-exclude .git/*',
            # include all LICENCE files
            '\n# include all license files found',
            'global-include %s/*LICENSE.*'%PACKAGE_NAME,
            # include all README files
            '\n# include all readme files found',
            'global-include %s/*README.*'%PACKAGE_NAME,
            'global-include %s/*readme.*'%PACKAGE_NAME]
with open('MANIFEST.in','w') as fd:
    for l in commands:
        fd.write(l)
        fd.write('\n')

# declare classifiers
CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU Affero General Public License v3
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Topic :: Software Development
Topic :: Software Development :: Build Tools
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

# create descriptions
LONG_DESCRIPTION = ["It's a Protein Data Bank (.pdb) files manipulation package that is mainly developed to parse and load, duplicate, manipulate and create pdb files.",
                    "A full description of a pdb file can be found here: http://deposit.rcsb.org/adit/docs/pdb_atom_format.html",
                    "pdbparser atoms configuration can be visualized by vmd software (http://www.ks.uiuc.edu/Research/vmd/) by simply pointing 'VMD_PATH' global variable to the exact path of vmd executable, and using 'visualize' method.",
                    "At any time and stage of data manipulation, a pdb file of all atoms or a subset of atoms can be exported to a pdb file."]
DESCRIPTION      = [ LONG_DESCRIPTION[0] ]

## get package info
PACKAGE_INFO={}
infoPath = convert_path('__pkginfo__.py')
with open(infoPath) as fd:
    exec(fd.read(), PACKAGE_INFO)


##############################################################################################
##################################### USEFUL DEFINITIONS #####################################
DATA_EXCLUDE = ('*.py', '*.pyc', '*~', '.*', '*.so', '*.pyd')
EXCLUDE_DIRECTORIES = ('*svn','*git','dist', 'EGG-INFO', '*.egg-info',)

def is_package(path):
    return (os.path.isdir(path) and os.path.isfile(os.path.join(path, '__init__.py')))

def get_packages(path, base="", exclude=None):
    if exclude is None:
        exclude = []
    assert isinstance(exclude, (list, set, tuple)), "exclude must be a list"
    exclude = [os.path.abspath(e) for e in exclude]
    packages = {}
    for item in os.listdir(path):
        d = os.path.join(path, item)
        if sum([e in os.path.abspath(d) for e in exclude]):
            continue
        if is_package(d):
            if base:
                module_name = "%(base)s.%(item)s" % vars()
            else:
                module_name = item
            packages[module_name] = d
            packages.update(get_packages(d, module_name, exclude))
    return packages

def find_package_data(where='.', package='', exclude=DATA_EXCLUDE,
                     exclude_directories=EXCLUDE_DIRECTORIES,
                     only_in_packages=True, show_ignored=False):
    out = {}
    stack = [(convert_path(where), '', package, only_in_packages)]
    while stack:
        where, prefix, package, only_in_packages = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where, name)
            if os.path.isdir(fn):
                bad_name = False
                for pattern in exclude_directories:
                    if (fnmatch.fnmatchcase(name, pattern)
                        or fn.lower() == pattern.lower()):
                        bad_name = True
                        if show_ignored:
                            print >> sys.stderr, ("Directory %s ignored by pattern %s" % (fn, pattern))
                        break
                if bad_name:
                    continue
                if (os.path.isfile(os.path.join(fn, '__init__.py')) and not prefix):
                    if not package:
                        new_package = name
                    else:
                        new_package = package + '.' + name
                    stack.append((fn, '', new_package, False))
                else:
                    stack.append((fn, prefix + name + '/', package, only_in_packages))
            elif package or not only_in_packages:
                # is a file
                bad_name = False
                for pattern in exclude:
                    if (fnmatch.fnmatchcase(name, pattern)
                        or fn.lower() == pattern.lower()):
                        bad_name = True
                        if show_ignored:
                            print >> sys.stderr, ("File %s ignored by pattern %s" % (fn, pattern))
                        break
                if bad_name:
                    continue
                out.setdefault(package, []).append(prefix+name)
    return out


def find_data(where=".", exclude=DATA_EXCLUDE, exclude_directories=EXCLUDE_DIRECTORIES, prefix=""):
    out = {}
    stack = [convert_path(where)]
    while stack:
        where = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where, name)
            d = os.path.join(prefix,os.path.dirname(fn))
            if os.path.isdir(fn):
                stack.append(fn)
            else:
                bad_name = False
                for pattern in exclude:
                    if (fnmatch.fnmatchcase(name, pattern) or fn.lower() == pattern.lower()):
                        bad_name = True
                        break
                if bad_name:
                    continue
                out.setdefault(d, []).append(fn)
    out = [(k,v) for k, v in out.items()]
    return out

################################## END OF USEFUL DEFINITIONS #################################
##############################################################################################


# get packages
PACKAGES = get_packages(path=PACKAGE_PATH, base='pdbparser',
                        exclude=(os.path.join(PACKAGE_NAME,"AMD"),
                                 os.path.join(PACKAGE_NAME,"docs")))
PACKAGES[PACKAGE_NAME] = '.'

# create meta data
metadata = dict(name = PACKAGE_NAME,
                packages=PACKAGES.keys(),
                package_dir=PACKAGES,
                version= PACKAGE_INFO['__version__'] ,
                author="Bachir AOUN",
                author_email="bachir.aoun@e-aoun.com",
                description = "\n".join(DESCRIPTION),
                long_description = "\n".join(LONG_DESCRIPTION),
                #url = "",
                #download_url = "",
                license = 'GNU',
                classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
                platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
                # Dependent packages (distributions)
                install_requires=['pysimplelog','pypref'], # it also needs numpy, but this is left for the user to install.
                setup_requires=[''],
                )

# setup
setup(**metadata)
