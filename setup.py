#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TAMkin is a post-processing toolkit for normal mode analysis, thermochemistry
# and reaction kinetics.
# Copyright (C) 2008-2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>, An Ghysels
# <An.Ghysels@UGent.be> and Matthias Vandichel <Matthias.Vandichel@UGent.be>
# Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all
# rights reserved unless otherwise stated.
#
# This file is part of TAMkin.
#
# TAMkin is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# In addition to the regulations of the GNU General Public License,
# publications and communications based in parts on this program or on
# parts of this program are required to cite the following article:
#
# "TAMkin: A Versatile Package for Vibrational Analysis and Chemical Kinetics",
# An Ghysels, Toon Verstraelen, Karen Hemelsoet, Michel Waroquier and Veronique
# Van Speybroeck, Journal of Chemical Information and Modeling, 2010, 50,
# 1736-1750W
# http://dx.doi.org/10.1021/ci100099g
#
# TAMkin is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--


from __future__ import print_function

import os
import subprocess
import sys

from setuptools import setup


# Try to get the version from git describe
__version__ = None
try:
    git_describe = subprocess.check_output(["git", "describe", "--tags"])
    version_words = git_describe.decode('utf-8').strip().split('-')
    __version__ = version_words[0]
    if len(version_words) > 1:
        __version__ += '.dev' + version_words[1]
except subprocess.CalledProcessError:
    pass

# Interact with version.py
fn_version = os.path.join(os.path.dirname(__file__), 'tamkin', 'version.py')
version_template = """\
\"""Do not edit this file, versioning is governed by ``git describe --tags`` and ``setup.py``.\"""
__version__ = '{}'
"""
if __version__ is None:
    # Try to load the git version tag from version.py
    try:
        with open(fn_version, 'r') as fh:
            __version__ = fh.read().split('=')[-1].replace('\'', '').strip()
    except IOError:
        print('Could not determine version. Giving up.')
        sys.exit(1)
else:
    # Store the git version tag in version.py
    with open(fn_version, 'w') as fh:
        fh.write(version_template.format(__version__))
        
        
setup(name='TAMkin',
    version=__version__,
    description='TAMkin is a post-processing toolkit for thermochemistry and kinetics analysis.',
    author='Toon Verstraelen, Matthias Vandichel, An Ghysels',
    author_email='Toon.Verstraelen@UGent.be, Matthias.Vandichel@UGent.be, An.Ghysels@UGent.be',
    url='http://molmod.ugent.be/code/',
    package_dir = {'tamkin': 'tamkin'},
    packages = ['tamkin', 'tamkin.io'],
    scripts=["scripts/tamkin-driver.py"],
    install_requires=['numpy>=1.0', 'nose>=0.11', 'matplotlib>1.1',
                      'molmod>=1.3.2', 'scipy>=0.17.1'],
    classifiers=[
        'Environment :: Console',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Intended Audience :: Science/Research',
    ],
)
