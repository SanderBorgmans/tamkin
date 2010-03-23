# TAMkin is a post-processing toolkit for thermochemistry and kinetics analysis.
# Copyright (C) 2008-2010 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
# Matthias Vandichel <Matthias.Vandichel@UGent.be> and
# An Ghysels <An.Ghysels@UGent.be>, Center for Molecular Modeling (CMM), Ghent
# University, Ghent, Belgium; all rights reserved unless otherwise stated.
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
# parts of this program are required to cite the following five articles:
#
# "Vibrational Modes in partially optimized molecular systems.", An Ghysels,
# Dimitri Van Neck, Veronique Van Speybroeck, Toon Verstraelen and Michel
# Waroquier, Journal of Chemical Physics, Vol. 126 (22): Art. No. 224102, 2007
# DOI:10.1063/1.2737444
#
# "Cartesian formulation of the Mobile Block Hesian Approach to vibrational
# analysis in partially optimized systems", An Ghysels, Dimitri Van Neck and
# Michel Waroquier, Journal of Chemical Physics, Vol. 127 (16), Art. No. 164108,
# 2007
# DOI:10.1063/1.2789429
#
# "Calculating reaction rates with partial Hessians: validation of the MBH
# approach", An Ghysels, Veronique Van Speybroeck, Toon Verstraelen, Dimitri Van
# Neck and Michel Waroquier, Journal of Chemical Theory and Computation, Vol. 4
# (4), 614-625, 2008
# DOI:10.1021/ct7002836
#
# "Mobile Block Hessian approach with linked blocks: an efficient approach for
# the calculation of frequencies in macromolecules", An Ghysels, Veronique Van
# Speybroeck, Ewald Pauwels, Dimitri Van Neck, Bernard R. Brooks and Michel
# Waroquier, Journal of Chemical Theory and Computation, Vol. 5 (5), 1203-1215,
# 2009
# DOI:10.1021/ct800489r
#
# "Normal modes for large molecules with arbitrary link constraints in the
# mobile block Hessian approach", An Ghysels, Dimitri Van Neck, Bernard R.
# Brooks, Veronique Van Speybroeck and Michel Waroquier, Journal of Chemical
# Physics, Vol. 130 (18), Art. No. 084107, 2009
# DOI:10.1063/1.3071261
#
# TAMkin is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


from tamkin.data import Molecule

from molmod import angstrom, amu, calorie, avogadro, lightspeed, centimeter
from molmod.periodic import periodic

import numpy


__all__ = [
    "load_molecule_charmm", "load_coordinates_charmm", "load_modes_charmm",
    "load_peptide_info_charmm",
]


def load_molecule_charmm(charmmfile_cor, charmmfile_hess,
                  is_periodic = False):
    """Units: CHARMM gradient in kcal/mol/angstrom, TAMkin internally all in atomic units

    Read from Hessian-CHARMM-file
    format:

    - nb of atoms = N
    - energy
    - gradient (N lines, 3 elements on each line)
    - Hessian  (upper triangular form, 1 element on each line)
    - coordinates (N lines, 3 elements on each line)
    """
    f = file(charmmfile_hess)

    N = int(f.readline().split()[-1])   # nb of atoms
    assert N > 0   # nb of atoms should be > 0

    energy = float(f.readline().split()[-1]) * 1000*calorie/avogadro

    gradient = numpy.zeros((N,3),float)
    for i,line in enumerate(f):
        words = line.split()
        gradient[i,:] = [float(word) for word in words]
        if i == (N-1):
            break
    gradient *= 1000*calorie/avogadro/angstrom
    hessian = numpy.zeros((3*N,3*N),float)
    row = 0
    col = 0
    for line in f:
        element = float(line.split()[-1])
        hessian[row,col]=element
        hessian[col,row]=element
        col += 1
        if col>=3*N:        #if this new col doesn't exist
            row += 1        #go to next row
            col = row       #to diagonal element
            if row >= 3*N:  #if this new row doesn't exist
               break
    hessian = hessian * 1000*calorie/avogadro /angstrom**2

    positions = numpy.zeros((N,3),float)
    for i,line in enumerate(f):
        words = line.split()
        positions[i,:] = [float(word)*angstrom for word in words]
        if i == (N-1):
            break
    f.close()

    # Read from coordinates-CHARMM-file
    # format:  header lines, which start with *
    #          N lines with   - mass in last column
    #                         - atomic type in 4th column
    f = file(charmmfile_cor)
    masses = numpy.zeros(N,float)
    symbols  = []
    for line in f:
        if not line.startswith("*"): # skip header lines
            break
    for i,line in enumerate(f):
        words = line.split()
        masses[i] = float(words[-1])*amu   # mass
        symbols.append( words[3] )         # symbol
        if i == (N-1):
            break
    f.close()

    # get corresponding atomic numbers
    mass_table = numpy.zeros(len(periodic))
    for i in xrange(1, len(mass_table)):
        m1 = periodic[i].mass
        if m1 is None:
            m1 = 200000.0
        m2 = periodic[i+1].mass
        if m2 is None:
            m2 = 200000.0
        mass_table[i] = 0.5*(m1+m2)
    atomicnumbers = numpy.zeros(N, int)
    for i,mass in enumerate(masses):
        atomicnumbers[i] = mass_table.searchsorted(mass)

    return Molecule(
        atomicnumbers, positions, masses, energy, gradient, hessian,
        1, # multiplicity
        symmetry_number=1, periodic=is_periodic, symbols=symbols
    )


def load_coordinates_charmm(filename):
    """Read coordinates from a standard CHARMM coordinate file

       Arguments:
         | filename  --  the CHARMM coordinate file (typically extension .crd or
                       .cor)

       Return values:
         | coordinates  --  coordinates in atomic units, numpy array with
                            shape (N,3)
         | masses  --  atomic masses in atomic units
         | symbols  --  list of CHARMM atom symbols
    """

    # skip the lines that start with * comments
    f = open(filename,'r')
    for line in f:
        if not line.startswith("*"): break
    N = int(line.split()[0])   # nb of atoms

    # store coordinates in Nbx3 matrix
    symbols = ['']*N
    coordinates = numpy.zeros((N,3),float)
    masses = numpy.zeros(N,float)
    count = 0
    for line in f:
        words = line.split()
        symbols[count]       = words[3]
        coordinates[count,:] = numpy.array([float(word) for word in words[4:7]])*angstrom
        masses[count]        = float(words[9])*amu
        count += 1
        if count >= N: break
    f.close()
    return coordinates, masses, symbols


def load_modes_charmm(filename):
    """Read modes, frequencies and masses from a standard CHARMM-modes-file

       The CHARMM-modes-file is generated by the VIBRAN module in CHARMM.

       Argument:
         | filename  --  CHARMM-modes-file

       Return values:
         | modes  --  numpy array with mass-weighted modes in the columns
         | freqs  --  numpy array with frequencies in atomic units
         | masses  --  atomic masses in atomic units
    """
    f = file(filename)

    # skip the lines that start with * comments
    for line in f:
        if not line.strip().startswith("*"): break

    # read nb of atoms and nbfreqs (if not yet specified by user)
    words = line.split()        # the current line does not start with a *
    nbfreqs = int(words[0])
    N       = int(words[1])/3   # nb of atoms

    # lines with masses, 6 masses on each line
    nblines = int(numpy.ceil(N/6.0))
    masses = numpy.zeros(N,float)
    count = 0
    for line in f:
        words = line.split()
        n = len(words)
        masses[count:count+n] = numpy.array([float(word) for word in words])
        count += n
        if count >= N: break

    # read nbfreqs freqs
    CNVFRQ = 2045.5/(2.99793*6.28319)  # conversion factor, see c36a0/source/fcm/consta.fcm in charmm code
    nblines = int(numpy.ceil(nbfreqs/6.0))
    freqs = numpy.zeros(nbfreqs, float)
    countline = 0
    countfreq = 0
    for line in f:
        words = line.split()
        for word in words:
            # do conversion
            freq_sq = float(word) #squared value
            if freq_sq > 0.0:  freq =  numpy.sqrt( freq_sq)
            else:              freq = -numpy.sqrt(-freq_sq) #actually imaginary
            freqs[countfreq] = freq * CNVFRQ * lightspeed/centimeter # conversion factor CHARMM, put into Tamkin internal units
            countfreq += 1
        countline += 1
        if countline >= nblines: break
    if countfreq != nbfreqs:
        raise ValueError("should have read "+str(nbfreqs)+" frequencies, but read "+str(countfreq))

    # read the nbfreqs modes
    modes = numpy.zeros((3*N,nbfreqs),float)
    row = 0
    col = 0
    for line in f:
        words = line.split()
        n = len(words)
        modes[row:row+n,col] = numpy.array([float(word) for word in words])
        row += n
        if row == 3*N:
            col += 1
            row = 0

    f.close()
    return modes, freqs, masses


def load_peptide_info_charmm(filename):
    """Load information from CHARMM file for peptide blocks and subsystems

       Arguments:
         | filename  --  the CHARMM coordinate file (typically extension .crd
                         or .cor)

       Return values:
         | N  --  total number of atoms in the peptide
         | calpha  --  indices of the alpha carbons ('CA' in CHARMM file)
         | proline  --  indices of the alpha carbons that belong to proline
                        residues ('PRO  CA' in CHARMM file)
         | carbon  -- indices of the backbone carbons, exclude the alpha carbons
                      ('C' in CHARMM file)
         | oxygen  -- indices of the backbone oxygens ('O' in CHARMM file)
         | nitrogen  --  indices of the backbone nitrogens ('N' in CHARMM file)
    """
    # Reading from charmmfile
    f = file(filename)
    # nb of atoms
    for i,line in enumerate(f):
        words = line.split()
        if words[0]!="*":
            N = int(words[0])
            break
    # find alpha carbons, proline residues, carbons, oxygens, nitrogens
    calpha = []
    proline = []
    carbon = []
    oxygen = []
    nitrogen = []
    for i,line in enumerate(f):
        words = line.split()
        if words[3].startswith("CA"):
            calpha.append(int(words[0]))
            if words[2]=="PRO":
                proline.append(int(words[0]))
        if words[3]=="C":
            carbon.append(int(words[0]))
        if words[3]=="O":
            oxygen.append(int(words[0]))
        if words[3]=="N":
            nitrogen.append(int(words[0]))
    f.close()
    return N, calpha, proline, carbon, oxygen, nitrogen



