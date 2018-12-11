# SOAP2_RawSpectra
SOAP 2.0: A Python package comprising the SOAP 2 stellar activity simulator, with proprietary modifications to output raw spectra.   
Original code by Xavier Dumusque.

*********************************************************************************
This folder includes:

* The soap2 Python package

* An create_spectra folder with data, a SOAP config file, an execution script
(soap_run.py) illustrating usage, and spot evolution configuration 
(spot_phase_config.py) and growth/decay function (spot_functions.py) files.

* 

The soap2 package is Xavier Dumusque's SOAP 2 distribution, with 
minor changes to make it a Python package (the original code was a
collection of pure python and extension modules that had to be in
the same folder as scripts using them).

 Heavy modifications made by Ari Silburt. Addititional modifications made by Christian Gilbertson to create more realistic sunspots and observations
*********************************************************************************
SOAP2 uses Python 2.  This release was tested in an Anaconda Python conda 
environment created and activated as follows:

$ conda create --name soap2 python=2.7 anaconda  
$ source activate soap2  

SOAP2 uses GSL 1 (last version: 1.16), from 2013.  Contemporary users
are likely to have a newer GSL in the default lib/include locations.
Linking against the right GSL requires platform-specific workarounds.
See below for Darwin (OS X, macOS) details.  This release has not
been tested on other platforms.

*********************************************************************************
To install soap2 into the Example directory (e.g., for testing), from within
the top soap2/ directory, run (must have a python 2 environment activated):

$ python setup.py install --install-lib=../Example

To run the example:

$ ipython -i soap_run.py  
or  
$ python soap_run.py  

It will produce several plots of SOAP-generated spectra. 

*********************************************************************************
The main changes vs. Xavier's version are:

* The stsp C extension module was renamed to _starspot, following common
Python naming conventions.

* The StSp Python module was renamed to ooapi; it provides a Python
object-oriented API for many methods in _starspot (but not all).

* Two includes in C extensions were changed to use quotes instead of
angle brackets (which should be reserved for std lib headers, and
raises an error with recent Darwin compilers).  The original lines:

starspot.c:#include <starspot.h>
starspotmodule.c:#include <starspot.h>

* A basic __init__.py was added to make the code collection a Python package,
including defining __all__ to enable "lazy" imports ("import *").

* The setup.py script (this file!) was revised to attempt to support 
installation with the out-of-date GSL version required by SOAP on Darwin
platforms (OS X, macOS).  This requires that the user install the proper
GSL and set environment variables pointing to the installation.

* Python code in the original distribution mixed tabs and spaces for 
indentation.  (Yikes!)  Python's tabnanny.py script reported that lines
in the Spot class in ooapi.py were ambiguous because of whitespace
(this did not cause any problems in test runs, however).  The whitespace
was fixed using Python's reindent.py script.  Visual inspection indicates
that reindent.py may add extra spaces in lines that were commented (so it
may not be clear how to uncomment them).  For reference, the original
files are included here, in the bad_tabs folder (as .py.bak files, backed
up by reindent.py).

* Spot evolution capabilities have beed added so that multiple spots can 
grow/decay simultaneously. Spot related parameters are specified in 
“Example/spot_phase_config.py”, while the growth/decay spot functions are
defined in “Example/spot_functions.py”.

* Naming conventions of output hdf5 files was changed to be more reflective 
of multiple spot cases. The information for each spot is stored as arrays in
the attributes of the output hdf5 file.

*********************************************************************************
Instructions for Darwin users:

Darwin users can use Homebrew to install GSL 1.  Homebrew puts outdated
libraries in the "keg" (a location for hidden packages), instructing users to
set environment variables to let the compiler and linker find them when keg
content should be preferred to content in standard locations.

As of 2017-05, the following commands install the last v1 version
of GSL and set the flags appropriately:

brew install gsl@1
export LDFLAGS=-L/usr/local/opt/gsl@1/lib
export CPPFLAGS=-I/usr/local/opt/gsl@1/include


Good luck starspotting!

- Tom Loredo, 2018-02-28
- Ari Silburt, 2018-03-13

