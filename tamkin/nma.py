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
# --
"""Normal mode analysis with default and extended schemes.

A normal mode analysis is carried out by constructing an NMA object. The first
argument is a molecule object created by one of the IO routines in
:mod:`tamkin.io`. ::

>>> nma = NMA(molecule)

This leads to a standard normal mode analysis in 3*N degrees of freedom.
The results, including those relevant for the construction of the molecular
partition function, are stored as attributes of the NMA object. For example::

>>> print nma.freqs

prints the frequencies of the normal modes. Note that all data is stored in
atomic units and that the freqs array contains really frequencies, not
wavenumbers. If you want to print the wavenumbers in cm**-1, use the unit
conversion constants from the ``molmod`` package::

>>> from molmod import centimeter, lightspeed
>>> invcm = lightspeed/centimeter
>>> print nma.freqs/invcm

One can also use modified schemes by giving a second argument to the NMA
constructor. The following example computes the normal modes in 3*N-6 degrees
of freedom::

>>> nma = NMA(molecule, ConstrainExt())

The second argument is an instance of a class that derives from the
:class:`Treatment` class. Other treatments include: :class:`Full` (the default),
:class:`PHVA`, :class:`VSA`, :class:`VSANoMass`, :class:`MBH`,
:class:`PHVA_MBH`, :class:`Constrain`, and :class:`MBHConstrainExt`.
"""

# A few conventions for the variables names:
#
# * The suffix _small refers to the small set of degrees of freedom that
#   obey constraints implicitly. The absence of the suffix refers to
#   conventional Cartesian coordinates, e.g. hessian is the 3Nx3N Hessian
#   and hessian_small is the Hessian in the new coordinates.
#
# * The suffix _mw stands for mass-weighted, e.g. hessian_mw is the mass
#   weighted Hessian in Cartesian coordinates. hessiam_small_mw is the mass
#   weighted Hessian in the new coordinates.


from tamkin.data import Molecule
from tamkin.io.internal import load_chk, dump_chk

import numpy as np


__all__ = [
    "NMA", "AtomDivision", "Transform", "MassMatrix", "Treatment",
    "Full"
]


class NMA(object):
    """A generic normal mode analysis class.

       This class gathers the functionality that is common between all types of
       NMA variations, i.e. computation of frequencies and modes, once the
       problem is transformed to reduced coordinates. The actual nature of the
       reduced coordinates is determined by the treatment argument.
    """

    def __init__(self, molecule, treatment=None, do_modes=True):
        """
           Arguments:
            | ``molecule`` -- a molecule object obtained from a routine in
                              :mod:`tamkin.io`

           Optional arguments:
            | ``treatment`` -- an instance of a Treatment subclass
                               [default=Full()]
            | ``do_modes`` -- When False, only the frequencies are computed.
                              When True, also the normal modes are computed.
                              [default=True]

           Referenced attributes of molecule:
              ``mass``, ``masses``, ``masses3``, ``numbers``, ``coordinates``,
              ``inertia_tensor``, ``multiplicity``, ``symmetry_number``,
              ``periodic``, ``energy``

           Extra attributes:
            | ``freqs`` -- array of frequencies
            | ``modes`` -- array of mass-weighted Cartesian modes (if do_modes
                           is True). Each column corresponds to one mode. One
                           has to divide a column the square root of the masses3
                           attribute to obtain the mode in non-mass-weighted
                           coordinates.
            | ``zeros`` -- list of indices of zero frequencies

        """
        treatment = Full()

        # the treatment object will store the results as attributes
        treatment(molecule, do_modes)
        # treatment.hessian_small:
        #    the Hessian in reduced coordinates
        # treatment.mass_matrix_small:
        #    the mass matrix in reduced coordinates (see MassMatrix class)
        # treatment.transform: (None if do_modes=False)
        #    the transformation from small displacements in reduced coordinates
        #    to small displacements in Cartesian coordinates. (see Transform class)
        # treatment.num_zeros:
        #    the number of zero eigenvalues to expect
        # treatment.external_basis: (None if do_modes=False)
        #    the basis of external degrees of freedom. number of basis vectors
        #    matches the number of zeros.
        #
        # For the implementation of certain treatments, it is easier to produce
        # a mass-weighted small Hessian immediately. In such cases, the
        # transform is also readily mass-weighted and mass_matrix_small is
        # None.

        # the conventional frequency computation in the reduced coordinates
        if treatment.mass_matrix_small is None:
            hessian_small_mw = treatment.hessian_small
        else:
            hessian_small_mw = treatment.mass_matrix_small.get_weighted_hessian(treatment.hessian_small)
        del treatment.hessian_small # save memory

        if hessian_small_mw.size == 0:
            self.freqs = np.array([])
            self.modes = np.array([])
            self.zeros = []
        else:
            if do_modes:
                evals, modes_small_mw = np.linalg.eigh(hessian_small_mw)
            else:
                evals = np.linalg.eigvalsh(hessian_small_mw)
                modes_small_mw = None

            # frequencies
            self.freqs = np.sqrt(abs(evals))/(2*np.pi)
            # turn imaginary frequencies into negative frequencies
            self.freqs *= (evals > 0)*2-1

            if do_modes:
                # At this point the transform object transforms unweighted reduced
                # coordinates into Cartesian coordinates. Now we will alter it, so that
                # it transforms from weighted reduced coordinates to Cartesian
                # coordinates.
                if treatment.mass_matrix_small is not None:
                    treatment.transform.make_weighted(treatment.mass_matrix_small)
                # transform the modes to unweighted Cartesian coordinates.
                self.modes = treatment.transform(modes_small_mw)
                # transform the modes to weighted Cartesian coordinates.
                self.modes *= molecule.masses3.reshape((-1,1))**0.5
            else:
                self.modes = None

            # guess which modes correspond to the zero frequencies
            if treatment.num_zeros == 0:
                # don't bother
                self.zeros = []
            else:
                if do_modes:
                    # take the 20 lowest modes and compute the overlap with the
                    # external basis
                    num_try = 20
                    to_try = abs(self.freqs).argsort()[:num_try]   #indices of lowest 20 modes
                    overlaps = np.zeros(num_try, float)
                    for counter, i in enumerate(to_try):
                        components = np.dot(treatment.external_basis, self.modes[:,i])
                        overlaps[counter] = np.linalg.norm(components)
                    self.zeros = to_try[overlaps.argsort()[-treatment.num_zeros:]]
                else:
                    self.zeros = abs(self.freqs).argsort()[:treatment.num_zeros]

        # a few more attributes that are worth keeping
        self.mass = molecule.mass
        self.masses = molecule.masses
        self.masses3 = molecule.masses3
        self.numbers = molecule.numbers
        self.coordinates = molecule.coordinates
        self.inertia_tensor = molecule.inertia_tensor
        self.multiplicity = molecule.multiplicity
        self.symmetry_number = molecule.symmetry_number
        self.periodic = molecule.periodic
        self.energy = molecule.energy
        self.title = molecule.title
        self.chemical_formula = molecule.chemical_formula

    def write_to_file(self, filename, fields='all'):
        """Write the NMA results to a human-readable checkpoint file.

           Argument:
            | ``filename`` -- the file to write to

           Optional argument:
            | ``fields`` -- define the selection of attributes to be written to
                            file. This is one of 'all' (all attributes), 'modes'
                            (only attributes required for nmatools.py), or
                            'partf' (only attributes required for the
                            construction of a partition function)
        """
        if fields == 'all':
            data = dict((key, val) for key, val in self.__dict__.items())
        elif fields == 'modes':
            keys = ["freqs", "modes", "masses", "numbers", "coordinates", "zeros", "title"]
            data = dict((key, self.__dict__[key]) for key in keys)
        elif fields == 'partf':
            keys = [
                "freqs", "mass", "masses3", "inertia_tensor", "multiplicity",
                "symmetry_number", "periodic", "energy", "zeros", "title",
                "chemical_formula",
            ]
            data = dict((key, self.__dict__[key]) for key in keys)
        dump_chk(filename, data)

    @classmethod
    def read_from_file(cls, filename):
        """Construct an NMA object from a previously saved checkpoint file

           Arguments:
            | ``filename`` -- the file to load from

           Usage::

             >>> nma = NMA.read_from_file("foo.chk")

        """
        # ugly way to bypass the default constructor
        result = cls.__new__(cls)
        # load the file
        data = load_chk(filename)
        # check the names of the fields:
        possible_fields = set([
            "freqs", "modes", "mass", "masses", "masses3", "numbers",
            "coordinates", "inertia_tensor", "multiplicity", "symmetry_number",
            "periodic", "energy", "zeros", "title", "chemical_formula",
        ])
        if not set(data.keys()).issubset(possible_fields):
            raise IOError("The Checkpoint file does not contain the correct fields.")
        # assign the attributes
        result.__dict__.update(data)
        return result


class AtomDivision(object):
    """A division of atoms into transformed, free and fixed."""

    def __init__(self, transformed, free, fixed):
        """
           Arguments:
            | ``transformed`` -- the atom indices of the atoms whose coordinates
                                 are transformed into non-Cartesian coordinates.
            | ``free`` -- the atom indices that are not transformed and retained
                          as Cartesian coordinates in the new set of coordinates
            | ``fixed`` -- the atoms that are not used for the new coordinates,
                           i.e. their positions are constrained.
        """
        self.transformed = np.array(transformed, int)
        self.free = np.array(free, int)
        self.fixed = np.array(fixed, int)

        self.num_cartesian = 3*(len(self.transformed)+len(self.free)+len(self.fixed))
        self.to_cartesian_order = np.zeros(self.num_cartesian, int)
        self.to_reduced_order = np.zeros(self.num_cartesian, int)
        counter = 0
        for l in self.transformed, self.free, self.fixed:
            for i in l:
                # index corresponds to Cartesian index
                # value corresponds to reduced index
                self.to_cartesian_order[3*i] = counter
                self.to_cartesian_order[3*i+1] = counter+1
                self.to_cartesian_order[3*i+2] = counter+2
                self.to_reduced_order[counter] = 3*i
                self.to_reduced_order[counter+1] = 3*i+1
                self.to_reduced_order[counter+2] = 3*i+2
                counter += 3


class Transform(object):
    """A clever transformation object. It is sparse when atom coordinates remain
       Cartesian in the reduced coordinates.

       This object transforms small displacements (first order) in reduced
       internal coordinates (can be mass weighted) into plain Cartesian
       coordinates.

       It is assumed that the reduced coordinates are always split into two
       parts (in order):

       1) the coordinates that are non-Cartesian
       2) the free coordinates that are Cartesian

    """

    def __init__(self, matrix, atom_division=None):
        """
           Arguments:
             | ``matrix`` -- the linear transformation from the transformed
                             displacements to Cartesian coordinates.

           Optional argument
             | ``atom_division`` -- an AtomDivision instance, when not given all
                                    atom coordinates are `transformed`

           Attributes:
             | ``matrix`` -- see above
             | ``scalars`` -- diagonal part of the linear transformation (only
                              used with mass-weighted transformations)
        """
        if matrix is None:
            matrix = np.zeros((0,0), float)
        if atom_division is None:
            # internal usage only:
            self._num_reduced = matrix.shape[1]
        else:
            # Quality Assurance:
            if matrix.shape[0] != 3*len(atom_division.transformed):
                raise ValueError("The matrix must have %i columns (matching the number of transformed atoms), got %i." %
                    3*len(atom_division.transformed), matrix.shape[0]
                )
            # internal usage only:
            self._num_reduced = matrix.shape[1] + 3*len(atom_division.free)

        self.matrix = matrix
        self.atom_division = atom_division
        # as long as we do not work with mass weighted coordinates, the
        # following remains None. In case of weighted coordinates, this
        # becomes a scaling vector with 3*len(free) floats:
        self.scalars = None
        self._weighted = False

    def get_weighted(self):
        """Return True when the transform is already mass-weighted"""
        return self._weighted

    weighted = property(get_weighted)

    def __call__(self, modes):
        """Transform small displacement vectors from new to Cartesian coordinates.

           Argument:
            | ``modes`` -- Small (mass-weighted) displacements (or modes) in
                           internal coordinates (float numpy array with shape
                           KxM, where K is the number of internal coordinates
                           and M is the number of modes)

           Returns:
              Small non-mass-weighted displacements (or modes) in Cartesian
              coordinates (float numpy array with shape 3NxM, where N is the
              number of Cartesian coordinates and M is the number of modes)

           Usage::

             >>> transform = Transform(...)
             >>> modes_cartesian = transform(modes_internal)

        """
        # Quality Assurance:
        if len(modes.shape) != 2:
            raise ValueError("Modes must be a two-dimensional array.")
        if modes.shape[0] != self._num_reduced:
            raise ValueError("The modes argument must be an array with %i rows, got %i." %
                (self._num_reduced, modes.shape[0])
            )
        # Computation
        if self.atom_division is None:
            return np.dot(self.matrix, modes)
        else:
            result = np.zeros((self.atom_division.num_cartesian, modes.shape[1]), float)  # 3NxM
            i1 = 3*len(self.atom_division.transformed)
            i2 = i1 + 3*len(self.atom_division.free)
            result[:i1] = np.dot(self.matrix, modes[:self.matrix.shape[1]])
            if self.weighted:
                result[i1:i2] = modes[self.matrix.shape[1]:]*self.scalars
            else:
                result[i1:i2] = modes[self.matrix.shape[1]:]
            #    result[:,i2:] remains zero because these atoms are fixed
            # Reorder the atoms and return the result
            tmp = result[self.atom_division.to_cartesian_order]
            return tmp

    def make_weighted(self, mass_matrix):
        """Include mass-weighting into the transformation.

           The original transformation is from non-mass-weighted new coordinates
           to non-mass-weighted Cartesian coordinates and becomes a transform
           from mass-weighted new coordinates to non-mass-weighted Cartesian
           coordinates.

           Argument:
            | ``mass_matrix`` -- A MassMatrix instance for the new coordinates
        """
        # modifies the transformation matrix in place:
        # the transformation matrix always transforms to non-mass-weighted Cartesian coords
        if self.weighted:
            raise Exception("The transformation is already weighted.")
        self.matrix = np.dot(self.matrix, mass_matrix.mass_block_inv_sqrt)
        self.scalars = mass_matrix.mass_diag_inv_sqrt.reshape((-1,1))
        self._weighted = True


class MassMatrix(object):
    """A clever mass matrix object. It is sparse when atom coordinates remain
       Cartesian in the reduced coordinates.
    """

    def __init__(self, *args):
        """
           Arguments, if one is given and it is a two-dimensional matrix:
            | ``mass_block`` -- the mass matrix associated with the transformed
                                coordinates

           Arguments, if one is given and it is a one-dimensional matrix:
            | ``mass_diag`` -- the diagonal of the mass matrix associated with
                               the free atoms (each mass appears three times)

           Arguments, if two are given:  ! Attention for order of arguments.
            | ``mass_block`` -- the mass matrix associated with the transformed
                                coordinates
            | ``mass_diag`` -- the diagonal of the mass matrix associated with
                               the free atoms (each mass appears three times)

           The mass of the fixed atoms does not really matter here.
        """
        if len(args) == 1:
            if len(args[0].shape) == 1:
                self.mass_diag = args[0]
                self.mass_block = np.zeros((0,0), float)
            elif len(args[0].shape) == 2:
                self.mass_diag = np.zeros((0,), float)
                self.mass_block = args[0]
            else:
                raise TypeError("When MassMatrix.__init__ gets one argument, it must be a one- or two-dimensional array.")
        elif len(args) == 2:
            self.mass_block = args[0]  #mass_block is first argument
            self.mass_diag  = args[1]  #mass_diag is second argument
        else:
            raise TypeError("MassMatrix.__init__ takes one or two arguments, %i given." % len(args))

        # the square root of the inverse
        if len(self.mass_block) == 0:
            self.mass_block_inv_sqrt = np.zeros((0,0), float)
        else:
            evals, evecs = np.linalg.eigh(self.mass_block)
            self.mass_block_inv_sqrt = np.dot(evecs/np.sqrt(evals), evecs.transpose())
        self.mass_diag_inv_sqrt = 1/np.sqrt(self.mass_diag)

    def get_weighted_hessian(self, hessian):
        hessian_mw = np.zeros(hessian.shape,float)
        n = len(self.mass_block)
        # transform block by block:
        hessian_mw[:n,:n] = np.dot(np.dot(self.mass_block_inv_sqrt, hessian[:n,:n]), self.mass_block_inv_sqrt)
        hessian_mw[:n,n:] = np.dot(self.mass_block_inv_sqrt, hessian[:n,n:])*self.mass_diag_inv_sqrt
        hessian_mw[n:,:n] = hessian[:n,n:].transpose()
        hessian_mw[n:,n:] = (hessian[n:,n:]*self.mass_diag_inv_sqrt).transpose()*self.mass_diag_inv_sqrt
        return hessian_mw


class Treatment(object):
    """An abstract base class for the NMA treatments. Derived classes must
       override the __call__ function, or they have to override the individual
       compute_zeros and compute_hessian methods. Parameters specific for the
       treatment are passed to the constructor, see for example the PHVA
       implementation.
    """

    def __init__(self):
        self.hessian_small = None
        self.mass_matrix_small = None
        self.transform = None
        self.num_zeros = None
        self.external_basis = None

    def __call__(self, molecule, do_modes):
        """Calls compute_hessian and compute_zeros (in order) with same arguments

           Arguments:
            | ``molecule`` -- a Molecule instance
            | ``do_modes`` -- a boolean indicates whether the modes have to be
                              computed
        """
        self.compute_hessian(molecule, do_modes)
        self.compute_zeros(molecule, do_modes)

    def compute_hessian(self, molecule, do_modes):
        """To be computed in derived classes

           Arguments:
            | ``molecule`` -- a Molecule instance
            | ``do_modes`` -- a boolean indicates whether the modes have to be

           Attributes to be computed:

           * ``treatment.hessian_small``: the Hessian in reduced coordinates
           * ``treatment.mass_matrix_small``: the mass matrix in reduced
             coordinates (see MassMatrix class)
           * ``treatment.transform``: (None if ``do_modes==False``) the
             transformation from small displacements in reduced coordinates
             to small displacements in Cartesian coordinates. (see Transform
             class)

           For the implementation of certain treatments, it is easier to produce
           a mass-weighted small Hessian immediately. In such cases, the
           transform is readily mass-weighted and mass_matrix_small is None.
        """
        raise NotImplementedError

    def compute_zeros(self, molecule, do_modes):
        """To be computed in derived classes

           Arguments:
            | ``molecule`` -- a Molecule instance
            | ``do_modes`` -- a boolean indicates whether the modes have to be

           Attributes to be computed:

           * ``treatment.num_zeros``: the number of zero eigenvalues to expect
           * ``treatment.external_basis``: (None if ``do_modes=False``) the
             basis of external degrees of freedom. number of basis vectors
             matches the number of zeros. These basis vectors are mass-weighted.
        """
        raise NotImplementedError


class Full(Treatment):
    """A full vibrational analysis, without transforming to a new set of
       coordinates.
    """
    def __init__(self, im_threshold=1.0):
        """
           Optional argument:
            | ``im_threshold`` -- Threshold for detection of deviations from
                                  linearity. When a moment of inertia is below
                                  this threshold, it is treated as a zero.
        """
        self.im_threshold = im_threshold
        Treatment.__init__(self)

    def compute_zeros(self, molecule, do_modes):
        """See :meth:`Treatment.compute_zeros`.

           The number of zeros should be:

           - 3 for a single atom, nonperiodic calculation
           - 5 for a linear molecule, nonperiodic calculation
           - 6 for a nonlinear molecule, nonperiodic calculation
           - 3 in periodic calculations
        """
        # determine nb of zeros
        external_basis = molecule.get_external_basis_new(self.im_threshold)
        self.num_zeros = external_basis.shape[0]

        # check
        if molecule.periodic:
            assert self.num_zeros == 3, "Number of zeros is expected to be 3 "\
                "(periodic calculation), but found %i." % self.num_zeros
        else:
            assert self.num_zeros in [3,5,6], "Number of zeros is expected to "\
                "be 3, 5 or 6, but found %i." % self.num_zeros

        if do_modes:
            # Mass-weighted and orthonormal basis vectors for external degrees
            # of freedom. These are used to detect which vibrational modes match
            # the external degrees of freedom.
            U, W, Vt = np.linalg.svd(molecule.get_external_basis_new(), full_matrices=False)
            self.external_basis = Vt

    def compute_hessian(self, molecule, do_modes):
        """See :meth:`Treatment.compute_hessian`.

        The Hessian is the full 3Nx3N Hessian matrix ``H``.
        The mass matrix is the full 3Nx3N mass matrix ``M``.
        It is assumed that the coordinates are Cartesian coordinates, so the
        mass matrix is diagonal.
        """
        self.hessian_small = molecule.hessian
        self.mass_matrix_small = MassMatrix(molecule.masses3)
        if do_modes:
            atom_division = AtomDivision([], np.arange(molecule.size), [])
            self.transform = Transform(None, atom_division)

