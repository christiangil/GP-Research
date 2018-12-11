#!/usr/bin/env python

"""
A Python package comprising the SOAP 2 stellar activity simulator.

This package is essentially Xavier Dumusque's SOAP 2 distribution, with 
minor changes to make it a Python package.  See the README for details
on the changes and installation advice.
"""

# To install the extensions in-place (i.e., in the source directory, with the
# pure Python modules), use:
#    python setup.py build_ext --inplace

import os, sys


include_dirs = ['.']
library_dirs = []

# SOAP2 uses GSL 1 (last version: 1.16), from 2013.  Contemporary users
# are likely to have a newer GSL in the default lib/include locations.

# Handle the outdated GSL dependency in an ugly platform-specific manner;
# consider SciSoft installations to be a "platform."

if os.path.exists('/usr/local/scisoft/'):
    include_dirs.append('/usr/local/scisoft/packages/gsl/include/')
    library_dirs.append('/usr/local/scisoft/packages/gsl/lib/')
elif sys.platform == 'linux2' or sys.platform == 'cygwin':
    include_dirs.extend(['/usr/local/include/', '/usr/include/'])
    library_dirs.extend(['/usr/lib64','/usr/lib/'])
elif sys.platform == 'darwin':
    if not os.environ.has_key('LDFLAGS'):
        raise RuntimeError('Follow Homebrew instructions for specifying linker location for GSL!')
    if not os.environ.has_key('CPPFLAGS'):
        raise RuntimeError('Follow Homebrew instructions for specifying compiler location for GSL!')
else:
    raise RuntimeError('Only Unix-like platforms are supported.  Sorry!')


def configuration(parent_package='',top_path=''):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('soap2', parent_package, top_path)

    config.add_extension('fit2',
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        libraries = ['gsl','gslcblas'],
        sources = ['soap2/fit2.c'])

    config.add_extension('_starspot',
        include_dirs = include_dirs,
        library_dirs = library_dirs,
        libraries = ['gsl','gslcblas'],
        sources      = ['soap2/starspotmodule.c','soap2/starspot.c'])

    config.add_extension('calculate_spectrum',
        include_dirs = include_dirs,
        libraries = ['m'],
        sources = ['soap2/compute_spectrum.c'])

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    #setup(**configuration(top_path='').todict())

    # author_email was xavier.bonfils@oal.ul.pt:
    # url was http://www.oal.ul.pt/~bonfils/:
    setup(configuration=configuration,
        name='soap2',
        version='0.1',
        description='Stellar activity simulator',
        author='X. Bonfils',
        author_email='xavier.bonfils@obs.ujf-grenoble.fr',  
        url='https://www.iau.org/administration/membership/individual/14852/',
        packages=['soap2'])
