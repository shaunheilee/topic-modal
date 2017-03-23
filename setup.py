#coding=utf8
# ========================================================
#   Copyright (C) 2017 All rights reserved.
#   
#   filename : setpu.py
#   author   : ***
#   date     : 2017-03-22
#   desc     : 
# ========================================================
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("gibbs", ["gibbs_sample.pyx"], include_dirs=[numpy.get_include()])]
setup(
name = 'gibbs sample module',
cmdclass = {'build_ext': build_ext},
ext_modules = ext_modules
)
