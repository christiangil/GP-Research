#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy, os

HOME = os.getenv("HOME")

#Change those variables to meet your system installation
macosx_python_include_dir = ['/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include/','/Library/Python/2.7/site-packages/numpy/core/include/']
linux_python_include_dir = ['/usr/lib64/python2.7/site-packages/numpy/core/include/','/usr/lib64/python2.6/site-packages/numpy/core/include/','/usr/lib/python2.3/site-packages/numpy/core/include/']

scisoft_gsl_include_dir = ['/usr/local/scisoft/packages/gsl/include/']
macport_gsl_include_dir = ['/opt/local/include/']
linux_gsl_include_dir = ['/usr/local/include/','/usr/include/']

scisoft_gsl_lib_dir = ['/usr/local/scisoft/packages/gsl/lib/']
macport_gsl_lib_dir = ['/opt/local/lib/','/opt/local/lib64/']
linux_gsl_lib_dir = ['/usr/lib64','/usr/lib/']

module1 = Extension('fit2',
                    include_dirs = ['.'] + macosx_python_include_dir + linux_python_include_dir + scisoft_gsl_include_dir + macport_gsl_include_dir + linux_gsl_include_dir,
                    library_dirs = ['/usr/local/lib','/opt/local/lib','/usr/local/scisoft/packages/gsl/lib'],
                    libraries = ['gsl','gslcblas'],
                    sources = ['fit2.c']
                    )

module2 = Extension('starspot_using_spectrum',
                    include_dirs = ['.'] + macosx_python_include_dir + linux_python_include_dir + scisoft_gsl_include_dir + macport_gsl_include_dir + linux_gsl_include_dir,
                    library_dirs = ['/usr/local/lib','/opt/local/lib','/usr/local/scisoft/packages/gsl/lib'],
                    libraries = ['gsl','gslcblas'],
                    sources      = ['starspotmodule_using_spectrum.c','starspot_using_spectrum.c']
                    )

module3 = Extension('calculate_spectrum',
        include_dirs = ['.'] + macosx_python_include_dir + linux_python_include_dir + scisoft_gsl_include_dir + macport_gsl_include_dir + linux_gsl_include_dir,
        libraries = ['m'],
        sources = ['compute_spectrum.c'])

setup(name='starspot',
      version='0.1',
      description='Star spot simulator',
      author='X. Bonfils',
      author_email='xavier.bonfils@oal.ul.pt',
      url='http://www.oal.ul.pt/~bonfils/',
      ext_modules=[module1, module2, module3])
