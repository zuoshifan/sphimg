"""
========================================================
Beam Transfer Matrices (:mod:`~sphimg.core.beamtransfer`)
========================================================

A class for calculating and managing Beam Transfer matrices

Classes
=======

.. autosummary::
    :toctree: generated/

    BeamTransfer

"""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import sys
import time

import numpy as np
import scipy.linalg as la
from sklearn.linear_model import BayesianRidge as br
import h5py

from sphimg.util import mpiutil, util, blockla
from sphimg.core import kltransform

from scalapy import core

from homotopy import homotopy


def complex_br(A, y, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=False, normalize=False, copy_A=True, verbose=True):
    """
    Minimization ||A x - y||^2 by using Bayesian ridge regression.

    Parameters
    ----------
    A : (M, N) ndarray
        Left matrix.
    y : (M,) ndarray
        Right hand vector.
    n_iter : int, optional
        Maximum number of iterations.  Default is 300.
    tol : float, optional
        Stop the algorithm if w has converged. Default is 1.e-3.
    alpha_1 : float, optional
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter. Default is 1.e-6
    alpha_2 : float, optional
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.
        Default is 1.e-6.
    lambda_1 : float, optional
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter. Default is 1.e-6.
    lambda_2 : float, optional
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.
        Default is 1.e-6
    compute_score : boolean, optional
        If True, compute the objective function at each step of the model.
        Default is False
    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
        Default is False.
    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
    copy_A : boolean, optional, default True
        If True, A will be copied; else, it may be overwritten.
    verbose : boolean, optional, default True
        Verbose mode when fitting the model.

    Returns
    -------
    x : (N,) ndarray
        The solution vector.
    """
    M, N = A.shape
    (M1,) = y.shape
    assert M == M1, 'Invalid shape for input matrix A and vector y: (%d, %d) and (%d)' % (M, N, M1)

    A1 = np.zeros((2*M, 2*N), dtype=A.real.dtype)
    A1[:M, :N] = A.real
    A1[:M, N:] = -A.imag
    A1[M:, :N] = A.imag
    A1[M:, N:] = A.real
    if not copy_A:
        del A
    y1 = np.zeros(2*M, dtype=y.real.dtype)
    y1[:M] = y.real
    y1[M:] = y.imag
    clf = br(n_iter=n_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2, compute_score=compute_score, fit_intercept=fit_intercept, normalize=normalize, copy_X=False, verbose=verbose)
    clf.fit(A1, y1)
    return clf.coef_[:N] + 1.0J * clf.coef_[N:]


def svd_gen(A, errmsg=None, *args, **kwargs):
    """Singular Value Decomposition of `A`.

    Factorizes the matrix `A` into two unitary matrices U and Vh, and
    a 1-D array s of singular values (real, non-negative) such that
    ``a == U*S*Vh``, where S is a suitably shaped matrix of zeros with
    main diagonal s.

    Parameters
    ----------
    A : (M, N) array_like
        Matrix to decompose.

    Returns
    -------
    U : ndarray
        Unitary matrix having left singular vectors as columns.
    s : ndarray
        The singular values, sorted in non-increasing order.
    Vh : ndarray
        Unitary matrix having right singular vectors as rows.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.
    """
    try:
        res = la.svd(A, *args, **kwargs)
    except la.LinAlgError:
        sv = la.svdvals(A)[0]
        At = A + sv * 1e-10 * np.eye(A.shape[0], A.shape[1])
        try:
            res = la.svd(At, *args, **kwargs)
        except la.LinAlgError as e:
            print "Failed completely. %s" % errmsg
            raise e

        if errmsg is None:
            print "Matrix SVD did not converge. Regularised."
        else:
            print "Matrix SVD did not converge (%s)." % errmsg

    return res


def matrix_image(A, rtol=1e-8, atol=None, errmsg=""):

    if A.shape[0] == 0:
        return np.array([], dtype=A.dtype).reshape(0, 0), np.array([], dtype=np.float64)

    try:
        # First try SVD to find matrix image
        u, s, v = la.svd(A, full_matrices=False)

        image, spectrum = u, s

    except la.LinAlgError as e:
        # Try QR with pivoting
        print "SVD1 not converged. %s" % errmsg

        q, r, p = la.qr(A, pivoting=True, mode='economic')

        try:
            # Try applying QR first, then SVD (this seems to help occasionally)
            u, s, v = la.svd(np.dot(q.T.conj(), A), full_matrices=False)

            image = np.dot(q, u)
            spectrum = s

        except la.LinAlgError as e:
            print "SVD2 not converged. %s" % errmsg

            image = q
            spectrum = np.abs(r.diagonal())

    if atol is None:
        cut = (spectrum > spectrum[0] * rtol).sum()
    else:
        cut = (spectrum > atol).sum()

    image = image[:, :cut].copy()

    return image, spectrum


def matrix_nullspace(A, rtol=1e-8, atol=None, errmsg=""):

    if A.shape[0] == 0:
        return np.array([], dtype=A.dtype).reshape(0, 0), np.array([], dtype=np.float64)

    try:
        # First try SVD to find matrix nullspace
        u, s, v = la.svd(A, full_matrices=True)

        nullspace, spectrum = u, s

    except la.LinAlgError as e:
        # Try QR with pivoting
        print "SVD1 not converged. %s" % errmsg

        q, r, p = la.qr(A, pivoting=True, mode='full')

        try:
            # Try applying QR first, then SVD (this seems to help occasionally)
            u, s, v = la.svd(np.dot(q.T.conj(), A))

            nullspace = np.dot(q, u)
            spectrum = s

        except la.LinAlgError as e:
            print "SVD2 not converged. %s" % errmsg

            nullspace = q
            spectrum = np.abs(r.diagonal())

    if atol is None:
        cut = (spectrum >= spectrum[0] * rtol).sum()
    else:
        cut = (spectrum >= atol).sum()

    nullspace = nullspace[:, cut:].copy()

    return nullspace, spectrum



class BeamTransfer(object):
    """A class for reading and writing Beam Transfer matrices from disk.

    In addition this provides methods for projecting vectors and matrices
    between the sky and the telescope basis.

    Parameters
    ----------
    directory : string
        Path of directory to read and write Beam Transfers from.
    telescope : sphimg.core.telescope.TransitTelescope, optional
        Telescope object to use for calculation. If `None` (default), try to
        load a cached version from the given directory.

    Attributes
    ----------
    svcut
    polsvcut
    ntel
    nsky
    nfreq
    svd_len
    ndofmax


    Methods
    -------
    ndof
    beam_m
    invbeam_m
    beam_svd
    beam_ut
    invbeam_svd
    beam_singularvalues
    generate
    project_vector_sky_to_telescope
    project_vector_telescope_to_sky
    project_vector_sky_to_svd
    project_vector_svd_to_sky
    project_vector_telescope_to_svd
    project_matrix_sky_to_telescope
    project_matrix_sky_to_svd
    """

    _mem_switch = 3.0 # Rough chunks (in GB) to divide calculation into.

    svdproj = True # Wether to do SVD projecton
    svcut = 1e-6
    polsvcut = 1e-4


    #====== Properties giving internal filenames =======

    @property
    def _picklefile(self):
        # The filename for the pickled telescope
        return self.directory + "/telescopeobject.pickle"

    @property
    def _tel_datafile(self):
        # File to save telescope frequencies and baselines
        return self.directory + '/telescope_data.hdf5'

    @property
    def _mdir(self):
        # Directory to save `m` ordered beam transfer matrix files
        return self.directory + '/beam_m/'

    def _mfile(self, mi):
        # Pattern to form the `m` ordered beam transfer matrix file
        pat = self._mdir + 'beam_%s.hdf5' % util.natpattern(self.telescope.mmax)
        return pat % abs(mi)

    @property
    def _ldir(self):
        # Directory to save `l` ordered beam transfer matrix files
        return self.directory + '/beam_l/'

    def _lfile(self, li):
        # Pattern to form the `l` ordered beam transfer matrix file
        pat = self._ldir + 'beam_%s.hdf5' % util.natpattern(self.telescope.lmax)
        return pat % abs(li)

    @property
    def _udir(self):
        # Directory to save `u` ordered beam transfer matrix files
        return self.directory + '/beam_u/'

    def _ufile(self, ui):
        # Pattern to form the `u` ordered beam transfer matrix file
        pat = self._udir + 'beam_%s.hdf5' % util.natpattern(self.telescope.u_max)
        return pat % abs(ui)

    # @property
    # def _fdir(self):
    #     # Pattern to form the `freq` ordered file.
    #     # pat = self.directory + "/beam_f/" + util.natpattern(self.telescope.nfreq)
    #     # return pat % fi
    #     return self.directory + "/beam_f/"

    # def _ffile(self, fi):
    #     # Pattern to form the `freq` ordered file.
    #     # return self._fdir(fi) + "/beam.hdf5"
    #     pat = self._fdir + 'beam_%s.hdf5' % util.natpattern(self.telescope.nfreq)
    #     return pat % fi

    @property
    def _svddir(self):
        # Directory to save `m` ordered svd beam files
        return self.directory + '/svd_m/'

    def _svdfile(self, mi):
        # Pattern to form the `m` ordered svd beam file
        pat = self._svddir + 'svd_%s.hdf5' % util.natpattern(self.telescope.mmax)
        return pat % abs(mi)

    @property
    def _svdspectrum_file(self):
        return self._svddir + 'svdspectrum.hdf5'

    #===================================================


    @property
    def _telescope_pickle(self):
        # The pickled telescope object
        return pickle.dumps(self.telescope)


    def __init__(self, directory, telescope=None, svdproj=True):

        self.directory = directory
        self.telescope = telescope
        self.svdproj = svdproj

        # Create directory if required
        if mpiutil.rank0 and not os.path.exists(directory):
            os.makedirs(directory)

        mpiutil.barrier()

        if self.telescope == None and mpiutil.rank0:
            print "Attempting to read telescope from disk..."

            try:
                f = open(self._picklefile, 'r')
                self.telescope = pickle.load(f)
            except IOError, UnpicklingError:
                raise Exception("Could not load Telescope object from disk.")


    #===================================================



    #====== Loading l-order beams ======================

    def _load_beam_l(self, li, fi=None):
        ## Read in beam from disk
        with h5py.File(self._lfile(li), 'r') as lfile:

            # If fi is None, return all frequency blocks. Otherwise just the one requested.
            if fi is None:
                beam = lfile['beam_l'][:]
            else:
                beam = lfile['beam_l'][fi][:]

        return beam


    @util.cache_last
    def beam_l(self, li, fi=None):
        """Fetch the meam transfer matrix for a given l.

        Parameters
        ----------
        li : integer
            l-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, npairs, npol_sky, 2*li+1)
        """

        return self._load_beam_l(li, fi=fi)

    #===================================================


    #====== Loading m-order beams ======================

    def _load_beam_m(self, mi, fi=None):
        ## Read in beam from disk
        with h5py.File(self._mfile(mi), 'r') as mfile:

            # If fi is None, return all frequency blocks. Otherwise just the one requested.
            if fi is None:
                beam = mfile['beam_m'][:]
            else:
                beam = mfile['beam_m'][fi][:]

        return beam


    @util.cache_last
    def beam_m(self, mi, fi=None):
        """Fetch the beam transfer matrix for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, 2, npairs, npol_sky, lmax+1)
        """

        return self._load_beam_m(mi, fi=fi)

    #===================================================


    #====== Loading l-order beams ======================

    def _load_beam_u(self, ui, fi=None):
        ## Read in beam from disk
        with h5py.File(self._ufile(ui), 'r') as ufile:

            # If fi is None, return all frequency blocks. Otherwise just the one requested.
            if fi is None:
                beam = ufile['beam_u'][:]
            else:
                beam = ufile['beam_u'][fi][:]

        return beam


    @util.cache_last
    def beam_u(self, ui, fi=None):
        """Fetch the meam transfer matrix for a given u.

        Parameters
        ----------
        ui : integer
            u-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, npairs, npol_sky, num_v)
        """

        return self._load_beam_u(ui, fi=fi)

    #===================================================


    #====== Loading freq-ordered beams =================

    # @util.cache_last
    # def _load_beam_freq(self, fi, fullm=False):

    #     tel = self.telescope
    #     mside = 2 * tel.lmax + 1 if fullm else 2 * tel.mmax + 1

    #     with h5py.File(self._ffile(fi), 'r') as ffile:
    #         beamf = ffile['beam_freq'][:]

    #     if fullm:
    #         beamt = np.zeros(beamf.shape[:-1] + (2*tel.lmax+1,), dtype=np.complex128)

    #         for mi in range(-tel.mmax, tel.mmax + 1):
    #             beamt[..., mi] = beamf[..., mi]

    #         beamf = beamt

    #     return beamf


    # @util.cache_last
    # def beam_freq(self, fi, fullm=False, single=False):
    #     """Fetch the beam transfer matrix for a given frequency.

    #     Parameters
    #     ----------
    #     fi : integer
    #         Frequency to fetch.
    #     fullm : boolean, optional
    #         Pad out m-modes such that we have :math:`mmax = 2*lmax-1`. Useful
    #         for projecting around a_lm's. Default is False.
    #     single : boolean, optional
    #         When set, fetch only the uncombined beam transfers (that is only
    #         positive or negative m). Default is False.

    #     Returns
    #     -------
    #     beam : np.ndarray
    #     """
    #     bf = self._load_beam_freq(fi, fullm)

    #     if single:
    #         return bf

    #     mside = (bf.shape[-1] + 1) / 2

    #     bfc = np.zeros((mside, 2) + bf.shape[:-1], dtype=bf.dtype)

    #     bfc[0, 0] = bf[..., 0]

    #     for mi in range(1, mside):
    #         bfc[mi, 0] = bf[..., mi]
    #         bfc[mi, 1] = (-1)**mi * bf[..., -mi].conj()

    #     return bfc

    #===================================================



    #====== Pseudo-inverse beams =======================

    noise_weight = True

    # @util.cache_last
    # def invbeam_m(self, mi):
    #     """Pseudo-inverse of the beam (for a given m).

    #     Uses the Moore-Penrose Pseudo-inverse as the optimal inverse for
    #     reconstructing the data. No `single` option as this only makes sense
    #     when combined.

    #     Parameters
    #     ----------
    #     mi : integer
    #         m-mode to calculate.

    #     Returns
    #     -------
    #     invbeam : np.ndarray (nfreq, npol_sky, lmax+1, 2, npairs)
    #     """

    #     beam = self.beam_m(mi)

    #     if self.noise_weight:
    #         noisew = self.telescope.noisepower(np.arange(self.telescope.npairs), 0).flatten()**(-0.5)
    #         beam = beam * noisew[:, np.newaxis, np.newaxis]

    #     beam = beam.reshape((self.nfreq, self.ntel, self.nsky))
    #     beamH = blockla.conj_dm(beam)
    #     BHB = blockla.multiply_dm_dm(beamH, beam)
    #     inv_BHB = blockla.inv_dm(BHB, hermi=True, rcond=1e-6)

    #     # ibeam = blockla.pinv_dm(beam, rcond=1e-6)
    #     ibeam = blockla.multiply_dm_dm(inv_BHB, beamH)

    #     if self.noise_weight:
    #         # Reshape to make it easy to multiply baselines by noise level
    #         ibeam = ibeam.reshape((-1, self.telescope.npairs))
    #         ibeam = ibeam * noisew

    #     shape = (self.nfreq, self.telescope.num_pol_sky,
    #              self.telescope.lmax + 1, self.ntel)

    #     return ibeam.reshape(shape)

    #===================================================




    #====== SVD Beam loading ===========================

    @util.cache_last
    def beam_svd(self, mi, fi=None):
        """Fetch the SVD beam transfer matrix (S V^H) for a given m. This SVD beam
        transfer projects from the sky into the SVD basis.

        This returns the full SVD spectrum. Cutting based on SVD value must be
        done by other routines (see project*svd methods).

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, svd_len, npol_sky, lmax+1)
        """

        if not self.svdproj:
            raise Exception('SVD beam transfer matrices not generated for self.svdproj == False.')

        with h5py.File(self._svdfile(mi), 'r') as svdfile:

            # Required array shape depends on whether we are returning all frequency blocks or not.
            if fi is None:
                bs = svdfile['beam_svd'][:]
            else:
                bs = svdfile['beam_svd'][fi][:]

        return bs


    # @util.cache_last
    # def invbeam_svd(self, mi, fi=None):
    #     """Fetch the SVD beam transfer matrix (S V^H) for a given m. This SVD beam
    #     transfer projects from the sky into the SVD basis.

    #     This returns the full SVD spectrum. Cutting based on SVD value must be
    #     done by other routines (see project*svd methods).

    #     Parameters
    #     ----------
    #     mi : integer
    #         m-mode to fetch.
    #     fi : integer
    #         frequency block to fetch. fi=None (default) returns all.

    #     Returns
    #     -------
    #     beam : np.ndarray (nfreq, svd_len, npol_sky, lmax+1)
    #     """

    #     svdfile = h5py.File(self._svdfile(mi), 'r')

    #     # Required array shape depends on whether we are returning all frequency blocks or not.
    #     if fi is None:
    #         ibs = svdfile['invbeam_svd'][:]
    #     else:
    #         ibs = svdfile['invbeam_svd'][fi][:]

    #     svdfile.close()

    #     return ibs


    @util.cache_last
    def beam_ut(self, mi, fi=None):
        """Fetch the SVD beam transfer matrix (U^H) for a given m. This SVD beam
        transfer projects from the telescope space into the SVD basis.

        This returns the full SVD spectrum. Cutting based on SVD value must be
        done by other routines (see project*svd methods).

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, svd_len, ntel)
        """

        if not self.svdproj:
            raise Exception('SVD beam U^H matrices not generated for self.svdproj == False.')

        with h5py.File(self._svdfile(mi), 'r') as svdfile:

            # Required array shape depends on whether we are returning all frequency blocks or not.
            if fi is None:
                bs = svdfile['beam_ut'][:]
            else:
                bs = svdfile['beam_ut'][fi][:]

        return bs


    @util.cache_last
    def beam_singularvalues(self, mi):
        """Fetch the vector of beam singular values for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.

        Returns
        -------
        beam : np.ndarray (nfreq, svd_len)
        """

        if not self.svdproj:
            raise Exception('SVD beam singularvalues not generated for self.svdproj == False.')

        with h5py.File(self._svdfile(mi), 'r') as svdfile:
            sv = svdfile['singularvalues'][:]

        return sv



    #===================================================



    #====== Generation of all the cache files ==========

    def generate(self, regen=False):
        """Save out all beam transfer matrices to disk.

        Parameters
        ----------
        regen : boolean, optional
            Force regeneration even if cache files exist (default: False).
        """

        self._generate_dirs()
        # self._generate_ffiles(regen)
        self._generate_teldatafile(regen)

        st = time.time()
        self._generate_mfiles(regen)
        self._generate_lfiles(regen)
        self._generate_ufiles(regen)
        et = time.time()
        if mpiutil.rank0:
            print "***** Beam transfer matrices generation time: %f" % (et - st)

        if self.svdproj:
            st = time.time()
            self._generate_svdfiles(regen)
            et = time.time()
            if mpiutil.rank0:
                print "***** SVD projection time: %f" % (et - st)
            # Collect the spectrum into a single file.
            self._collect_svd_spectrum(regen)

        # Save pickled telescope object
        if mpiutil.rank0:
            print
            print '=' * 80
            print "=== Saving Telescope object. ==="
            with open(self._picklefile, 'w') as f:
                pickle.dump(self.telescope, f)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()


    generate_cache = generate # For compatibility with old code


    def _generate_dirs(self):
        ## Create all the directories required to store the beam transfers.

        if mpiutil.rank0:

            # Create main directory for beamtransfer
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            # Create directories for storing frequency ordered beams
            # if not os.path.exists(self._fdir):
            #     os.makedirs(self._fdir)

            # Create directories for m beams
            if not os.path.exists(self._mdir):
                os.makedirs(self._mdir)

            # Create directories for l beams
            if not os.path.exists(self._ldir):
                os.makedirs(self._ldir)

            # Create directories for u beams
            if not os.path.exists(self._udir):
                os.makedirs(self._udir)

            # Create directories for svd files if need to do SVD projection
            if not os.path.exists(self._svddir) and self.svdproj:
                os.makedirs(self._svddir)

        mpiutil.barrier()


    # def _generate_ffiles(self, regen=False):
    #     ## Generate the beam transfers ordered by frequency.
    #     ## Divide frequencies between MPI processes and calculate the beams
    #     ## for the baselines, then write out into separate files.

    #     for fi in mpiutil.mpirange(self.nfreq):

    #         if os.path.exists(self._ffile(fi)) and not regen:
    #             print ("f index %i. File: %s exists. Skipping..." %
    #                    (fi, (self._ffile(fi))))
    #             continue
    #         else:
    #             print ('f index %i. Creating file: %s' %
    #                    (fi, (self._ffile(fi))))

    #         f = h5py.File(self._ffile(fi), 'w')

    #         # Set a few useful attributes.
    #         # f.attrs['baselines'] = self.telescope.baselines
    #         # f.attrs['baseline_indices'] = np.arange(self.telescope.npairs)
    #         f.attrs['frequency_index'] = fi
    #         f.attrs['frequency'] = self.telescope.frequencies[fi]
    #         f.attrs['cylobj'] = self._telescope_pickle

    #         dsize = (self.telescope.nbase, self.telescope.num_pol_sky, self.telescope.lmax+1, 2*self.telescope.mmax+1)

    #         csize = (min(10, self.telescope.nbase), self.telescope.num_pol_sky, self.telescope.lmax+1, 1)

    #         dset = f.create_dataset('beam_freq', dsize, chunks=csize, compression='lzf', dtype=np.complex128)

    #         # Divide into roughly 5 GB chunks
    #         nsections = int(np.ceil(np.prod(dsize) * 16.0 / 2**30.0 / self._mem_switch))

    #         print "Dividing calculation of %f GB array into %i sections." % (np.prod(dsize) * 16.0 / 2**30.0, nsections)

    #         b_sec = np.array_split(np.arange(self.telescope.npairs, dtype=np.int), nsections)
    #         f_sec = np.array_split(fi * np.ones(self.telescope.npairs, dtype=np.int), nsections)

    #         # Iterate over each section, generating transfers and save them.
    #         for si in range(nsections):
    #             print "Calculating section %i of %i...." % (si, nsections)
    #             b_ind, f_ind = b_sec[si], f_sec[si]
    #             tarray = self.telescope.transfer_matrices(b_ind, f_ind)
    #             dset[(b_ind[0]):(b_ind[-1]+1), ..., :(self.telescope.mmax+1)] = tarray[..., :(self.telescope.mmax+1)]
    #             dset[(b_ind[0]):(b_ind[-1]+1), ..., (-self.telescope.mmax):]  = tarray[..., (-self.telescope.mmax):]
    #             del tarray

    #         f.close()

    #     # If we're part of an MPI run, synchronise here.
    #     mpiutil.barrier()


    def _generate_teldatafile(self, regen=False):

        if mpiutil.rank0:
            if os.path.exists(self._tel_datafile) and not regen:
                print
                print '=' * 80
                print 'File %s exists. Skipping...' % self._tel_datafile
            else:
                print
                print '=' * 80
                print 'Crreate telescope data file %s...' % self._tel_datafile
                with h5py.File(self._tel_datafile, 'w') as f:
                    f.create_dataset('baselines', data=self.telescope.baselines)
                    f.create_dataset('frequencies', data=self.telescope.frequencies)

        mpiutil.barrier()


    def _generate_mfiles(self, regen=False):

        completed_file = self._mdir + 'COMPLETED_BEAM'
        if os.path.exists(completed_file) and not regen:
            if mpiutil.rank0:
                print
                print '=' * 80
                print "******* Beam transfer m-files already generated ********"
            mpiutil.barrier()
            return

        if mpiutil.rank0:
            print
            print '=' * 80
            print 'Create beam transfer m-files...'

        st = time.time()

        nfb = self.telescope.nfreq * self.telescope.nbase
        fbmap = np.mgrid[:self.telescope.nfreq, :self.telescope.nbase].reshape(2, nfb)

        # Calculate the number of baselines to deal with at any one time. Aim
        # to have a maximum of `self._mem_switch` GB in memory at any one time
        fbsize = self.telescope.num_pol_sky * (self.telescope.lmax+1) * (2*self.telescope.mmax+1) * 16.0

        nodemem = self._mem_switch * 2**30.0

        num_fb_per_node = int(nodemem / fbsize)
        num_fb_per_chunk = num_fb_per_node * mpiutil.size
        num_chunks = int(np.ceil(1.0 * nfb / num_fb_per_chunk))  # Number of chunks to break the calculation into

        if mpiutil.rank0:
            print "Splitting into %i chunks...." % num_chunks

        # The local m sections
        lm, sm, em = mpiutil.split_local(self.telescope.mmax+1)

        # Iterate over all m's and create the hdf5 files we will write into.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            # don't save all zero values for `l` < `m` to save disk space
            dsize = (self.telescope.nfreq, 2, self.telescope.nbase, self.telescope.num_pol_sky, self.telescope.lmax+1 - mi)
            csize = (1, 2, min(10, self.telescope.nbase), self.telescope.num_pol_sky, self.telescope.lmax+1 - mi)

            with h5py.File(self._mfile(mi), 'w') as f:
                f.create_dataset('beam_m', dsize, chunks=csize, compression='lzf', dtype=np.complex128)
                # Write useful attributes.
                f.attrs['m'] = mi

        mpiutil.barrier()

        # Iterate over chunks
        for ci, fbrange in enumerate(mpiutil.split_m(nfb, num_chunks).T):

            if mpiutil.rank0:
                print "Starting chunk %i of %i" % (ci+1, num_chunks)

            # Unpack freq-baselines range into num, start and end
            fbnum, fbstart, fbend = fbrange

            # Split the fb list into the ones local to this node
            loc_num, loc_start, loc_end = mpiutil.split_local(fbnum)
            fb_ind = range(fbstart + loc_start, fbstart + loc_end)

            # Extract the local frequency and baselines indices
            f_ind = fbmap[0, fb_ind]
            bl_ind = fbmap[1, fb_ind]

            # Create array to hold local matrix section
            fb_array = np.zeros((loc_num, 2, self.telescope.num_pol_sky, self.telescope.lmax+1, self.telescope.mmax+1), dtype=np.complex128)

            if loc_num > 0:

                # Calculate the local Beam Matrices
                tarray = self.telescope.transfer_matrices(bl_ind, f_ind, centered=False)

                # Expensive memory copy into array section
                for mi in range(1, self.telescope.mmax+1):
                    fb_array[:, 0, ..., mi] = tarray[..., mi]
                    fb_array[:, 1, ..., mi] = (-1)**mi * tarray[..., -mi].conj()

                fb_array[:, 0, ..., 0] = tarray[..., 0]

                del tarray

            if mpiutil.rank0:
                print "Transposing and writing chunk."

            # Perform an in memory MPI transpose to get the m-ordered array
            m_array = mpiutil.transpose_blocks(fb_array, (fbnum, 2, self.telescope.num_pol_sky, self.telescope.lmax + 1, self.telescope.mmax + 1))

            del fb_array

            # Write out the current set of chunks into the m-files.
            for lmi, mi in enumerate(range(sm, em)):

                # Open up correct m-file
                with h5py.File(self._mfile(mi), 'r+') as mfile:

                    # Lookup where to write Beam Transfers and write into file.
                    for fbl, fbi in enumerate(range(fbstart, fbend)):
                        fi = fbmap[0, fbi]
                        bi = fbmap[1, fbi]
                        # mfile['beam_m'][fi, :, bi] = m_array[fbl, ..., lmi]
                        mfile['beam_m'][fi, :, bi] = m_array[fbl, ..., mi:, lmi] # noly save l >= m

            del m_array

        mpiutil.barrier()

        et = time.time()

        if mpiutil.rank0:
            # Make file marker that the m's have been correctly generated:
            open(completed_file, 'a').close()

            # Print out timing
            print "=== MPI transpose took %f s ===" % (et - st)

        mpiutil.barrier()


    def _generate_lfiles(self, regen=False):

        completed_file = self._ldir + 'COMPLETED_BEAM'
        if os.path.exists(completed_file) and not regen:
            if mpiutil.rank0:
                print
                print '=' * 80
                print "******* Beam transfer l-files already generated ********"
            mpiutil.barrier()
            return

        if mpiutil.rank0:
            print
            print '=' * 80
            print 'Create beam transfer l-files...'

        st = time.time()

        nfb = self.telescope.nfreq * self.telescope.nbase
        fbmap = np.mgrid[:self.telescope.nfreq, :self.telescope.nbase].reshape(2, nfb)

        # Calculate the number of baselines to deal with at any one time. Aim
        # to have a maximum of `self._mem_switch` GB in memory at any one time
        lmax = self.telescope.lmax
        fbsize = self.telescope.num_pol_sky * (lmax+1)**2 * 16.0

        nodemem = self._mem_switch * 2**30.0

        num_fb_per_node = int(nodemem / fbsize)
        num_fb_per_chunk = num_fb_per_node * mpiutil.size
        num_chunks = int(np.ceil(1.0 * nfb / num_fb_per_chunk))  # Number of chunks to break the calculation into

        if mpiutil.rank0:
            print "Splitting into %i chunks...." % num_chunks

        # The local l sections
        ll, sl, el = mpiutil.split_local(lmax+1)

        # Iterate over all l's and create the hdf5 files we will write into.
        for li in mpiutil.mpirange(lmax + 1):

            dsize = (self.telescope.nfreq, self.telescope.nbase, self.telescope.num_pol_sky, 2*li + 1)
            csize = (1, min(10, self.telescope.nbase), self.telescope.num_pol_sky, 2*li + 1)

            with h5py.File(self._lfile(li), 'w') as f:
                f.create_dataset('beam_l', dsize, chunks=csize, compression='lzf', dtype=np.complex128)
                # Write useful attributes.
                f.attrs['l'] = li

        mpiutil.barrier()

        # Iterate over chunks
        for ci, fbrange in enumerate(mpiutil.split_m(nfb, num_chunks).T):

            if mpiutil.rank0:
                print "Starting chunk %i of %i" % (ci+1, num_chunks)

            # Unpack freq-baselines range into num, start and end
            fbnum, fbstart, fbend = fbrange

            # Split the fb list into the ones local to this node
            loc_num, loc_start, loc_end = mpiutil.split_local(fbnum)
            fb_ind = range(fbstart + loc_start, fbstart + loc_end)

            # Extract the local frequency and baselines indices
            f_ind = fbmap[0, fb_ind]
            bl_ind = fbmap[1, fb_ind]

            # Create array to hold local matrix section, `l` axes is the last axes inorder to do the following transpose_blocks
            fb_array = np.zeros((loc_num, self.telescope.num_pol_sky, 2*lmax+1, lmax+1), dtype=np.complex128)

            if loc_num > 0:

                # Calculate the local Beam Matrices
                tarray = self.telescope.transfer_matrices(bl_ind, f_ind, centered=True)

                # Expensive memory copy into array section
                for li in range(1, lmax+1):
                    fb_array[..., li] = tarray[..., li, :]

                del tarray

            if mpiutil.rank0:
                print "Transposing and writing chunk."

            # Perform an in memory MPI transpose to get the l-ordered array
            # This makes the original data distributed along chunks to redistributes along different `l`s, i.e., different `l`s now distribute on different processes
            print 'fb_array.shape: ', fb_array.shape
            l_array = mpiutil.transpose_blocks(fb_array, (fbnum, self.telescope.num_pol_sky, 2*lmax + 1, lmax + 1))
            print 'l_array.shape: ', l_array.shape

            del fb_array

            # Write out the current set of chunks into the m-files.
            for lli, li in enumerate(range(sl, el)):

                # Open up correct l-file
                with h5py.File(self._lfile(li), 'r+') as lfile:

                    # Lookup where to write Beam Transfers and write into file.
                    for fbl, fbi in enumerate(range(fbstart, fbend)):
                        fi = fbmap[0, fbi]
                        bi = fbmap[1, fbi]
                        lfile['beam_l'][fi, bi] = l_array[fbl, :, lmax-li:lmax+li+1, lli]

            del l_array

        mpiutil.barrier()

        et = time.time()

        if mpiutil.rank0:
            # Make file marker that the m's have been correctly generated:
            open(completed_file, 'a').close()

            # Print out timing
            print "=== MPI transpose took %f s ===" % (et - st)

        mpiutil.barrier()


    def _generate_ufiles(self, regen=False):

        completed_file = self._udir + 'COMPLETED_BEAM'
        if os.path.exists(completed_file) and not regen:
            if mpiutil.rank0:
                print
                print '=' * 80
                print "******* Beam transfer u-files already generated ********"
            mpiutil.barrier()
            return

        if mpiutil.rank0:
            print
            print '=' * 80
            print 'Create beam transfer u-files...'

        st = time.time()

        nfb = self.telescope.nfreq * self.telescope.nbase
        fbmap = np.mgrid[:self.telescope.nfreq, :self.telescope.nbase].reshape(2, nfb)

        # Split the fb list into the ones local to this node
        loc_num, loc_start, loc_end = mpiutil.split_local(nfb)
        fb_ind = range(loc_start, loc_end)

        u_max = self.telescope.u_max
        v_max = self.telescope.v_max

        # Calculate the number of baselines to deal with at any one time. Aim
        # to have a maximum of `self._mem_switch` GB in memory at any one time
        fbsize = self.telescope.num_pol_sky * (2*u_max+1) * (2*v_max+1) * 16.0

        nodemem = self._mem_switch * 2**30.0

        num_fb_per_node = int(nodemem / fbsize)
        num_fb_per_chunk = num_fb_per_node * mpiutil.size
        num_chunks = int(np.ceil(1.0 * nfb / num_fb_per_chunk))  # Number of chunks to break the calculation into

        if mpiutil.rank0:
            print "Splitting into %i chunks...." % num_chunks

        # The local m sections
        lu, su, eu = mpiutil.split_local(u_max+1)

        # Iterate over all u's and create the hdf5 files we will write into.
        for ui in mpiutil.mpirange(u_max+1):

            dsize = (self.telescope.nfreq, 2, self.telescope.nbase, self.telescope.num_pol_sky, 2*v_max+1)
            csize = (1, 2, min(10, self.telescope.nbase), self.telescope.num_pol_sky, 2*v_max+1)

            with h5py.File(self._ufile(ui), 'w') as f:
                f.create_dataset('beam_u', dsize, chunks=csize, compression='lzf', dtype=np.complex128)
                # Write useful attributes.
                f.attrs['u'] = ui

        mpiutil.barrier()

        # Iterate over chunks
        for ci, fbrange in enumerate(mpiutil.split_m(nfb, num_chunks).T):

            if mpiutil.rank0:
                print "Starting chunk %i of %i" % (ci+1, num_chunks)

            # Unpack freq-baselines range into num, start and end
            fbnum, fbstart, fbend = fbrange

            # Split the fb list into the ones local to this node
            loc_num, loc_start, loc_end = mpiutil.split_local(fbnum)
            fb_ind = range(fbstart + loc_start, fbstart + loc_end)

            # Extract the local frequency and baselines indices
            f_ind = fbmap[0, fb_ind]
            bl_ind = fbmap[1, fb_ind]

            # Create array to hold local matrix section
            fb_array = np.zeros((loc_num, 2, self.telescope.num_pol_sky, 2*v_max+1, u_max+1), dtype=np.complex128)

            if loc_num > 0:

                # Calculate the local Beam Matrices
                tarray = self.telescope.transfer_uv(bl_ind, f_ind)

                # Expensive memory copy into array section
                for ui in range(1, u_max+1):
                    fb_array[:, 0, ..., ui] = tarray[..., ui, :]
                    fb_array[:, 1, ..., ui] = tarray[..., -ui, :]

                fb_array[:, 0, ..., 0] = tarray[..., 0, :]

                del tarray

            if mpiutil.rank0:
                print "Transposing and writing chunk."

            # Perform an in memory MPI transpose to get the u-ordered array
            u_array = mpiutil.transpose_blocks(fb_array, (fbnum, 2, self.telescope.num_pol_sky, 2*v_max + 1, u_max + 1))

            del fb_array

            # Write out the current set of chunks into the u-files.
            for lui, ui in enumerate(range(su, eu)):

                # Open up correct u-file
                with h5py.File(self._ufile(ui), 'r+') as ufile:

                    # Lookup where to write Beam Transfers and write into file.
                    for fbl, fbi in enumerate(range(fbstart, fbend)):
                        fi = fbmap[0, fbi]
                        bi = fbmap[1, fbi]
                        ufile['beam_u'][fi, :, bi] = u_array[fbl, ..., lui]

            del u_array

        mpiutil.barrier()

        et = time.time()

        if mpiutil.rank0:
            # Make file marker that the m's have been correctly generated:
            open(completed_file, 'a').close()

            # Print out timing
            print "=== MPI transpose took %f s ===" % (et - st)

        mpiutil.barrier()


    def _generate_svdfiles(self, regen=False):
        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available.

        completed_file = self._svddir + 'COMPLETED_SVD'
        if os.path.exists(completed_file) and not regen:
            if mpiutil.rank0:
                print
                print '=' * 80
                print "******* Beam transfer svd-files already generated ********"
            mpiutil.barrier()
            return

        if mpiutil.rank0:
            print
            print '=' * 80
            print 'Create beam transfer svd-files...'

        # Data shape for the stokes T U-matrix (left evecs)
        dsize_ut = (self.telescope.nfreq, self.svd_len, self.ntel)
        csize_ut = (1, min(10, self.svd_len), self.ntel)

        # Data shape for the singular values.
        dsize_sig = (self.telescope.nfreq, self.svd_len)

        # mpiutil.barrier()

        # print 'Process %d starting svd file...' % mpiutil.rank
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._svdfile(mi)) and not regen:
                print "File %s exists. Skipping..." % self._svdfile(mi)
                continue
            # else:
            #     print 'Creating SVD file: %s' % self._svdfile(mi)

            # Open m beams for reading and file to write SVD results into.
            with h5py.File(self._mfile(mi), 'r') as fm, h5py.File(self._svdfile(mi), 'w') as fs:

                # Create a chunked dataset for writing the SVD beam matrix into.
                dsize_bsvd = (self.telescope.nfreq, self.svd_len, self.telescope.num_pol_sky, self.telescope.lmax+1 - mi)
                csize_bsvd = (1, min(10, self.svd_len), self.telescope.num_pol_sky, self.telescope.lmax+1 - mi)
                dset_bsvd = fs.create_dataset('beam_svd', dsize_bsvd, chunks=csize_bsvd, compression='lzf', dtype=np.complex128)

                dset_ut  = fs.create_dataset('beam_ut', dsize_ut, chunks=csize_ut, compression='lzf', dtype=np.complex128)

                # Create a dataset for the singular values.
                dset_sig  = fs.create_dataset('singularvalues', dsize_sig, dtype=np.float64)

                ## For each frequency in the m-files read in the block, SVD it,
                ## and construct the new beam matrix, and save.
                for fi in np.arange(self.telescope.nfreq):

                    # Read the positive and negative m beams, and combine into one.
                    bf = fm['beam_m'][fi][:].reshape(self.ntel, self.telescope.num_pol_sky, self.telescope.lmax+1 - mi)

                    noisew = self.telescope.noisepower(np.arange(self.telescope.npairs), fi).flatten()**(-0.5)
                    noisew = np.concatenate([noisew, noisew])
                    bf = bf * noisew[:, np.newaxis, np.newaxis]

                    # Reshape total beam to a 2D matrix
                    bfr = bf.reshape(self.ntel, -1)

                    # If unpolarised skip straight to the final SVD, otherwise
                    # project onto the polarised null space.
                    if self.telescope.num_pol_sky == 1:
                        bf2 = bfr
                        ut2 = np.identity(self.ntel, dtype=np.complex128)
                    else:
                        ## SVD 1 - coarse projection onto sky-modes
                        u1, s1 = matrix_image(bfr, rtol=1e-10, errmsg=("SVD1 m=%i f=%i" % (mi, fi)))

                        ut1 = u1.T.conj()
                        bf1 = np.dot(ut1, bfr)

                        ## SVD 2 - project onto polarisation null space
                        bfp = bf1.reshape(bf1.shape[0], self.telescope.num_pol_sky, self.telescope.lmax+1 - mi)[:, 1:]
                        bfp = bfp.reshape(bf1.shape[0], (self.telescope.num_pol_sky - 1) * (self.telescope.lmax+1 - mi))
                        u2, s2 = matrix_nullspace(bfp, rtol=self.polsvcut, errmsg=("SVD2 m=%i f=%i" % (mi, fi)))

                        ut2 = np.dot(u2.T.conj(), ut1)
                        bf2 = np.dot(ut2, bfr)

                    # Check to ensure polcut hasn't thrown away all modes. If it
                    # has, just leave datasets blank.
                    if bf2.shape[0] > 0 and (self.telescope.num_pol_sky == 1 or (s1 > 0.0).any()):

                        ## SVD 3 - decompose polarisation null space
                        bft = bf2.reshape(-1, self.telescope.num_pol_sky, self.telescope.lmax+1 - mi)[:, 0]

                        u3, s3 = matrix_image(bft, rtol=0.0, errmsg=("SVD3 m=%i f=%i" % (mi, fi)))
                        ut3 = np.dot(u3.T.conj(), ut2)

                        nmodes = ut3.shape[0]

                        # Skip if nmodes is zero for some reason.
                        if nmodes == 0:
                            continue

                        # Final products
                        ut = ut3
                        sig = s3[:nmodes]
                        beam = np.dot(ut3, bfr)
                        # ibeam = la.pinv(beam)

                        # Save out the evecs (for transforming from the telescope frame into the SVD basis)
                        dset_ut[fi, :nmodes] = (ut * noisew[np.newaxis, :])

                        # Save out the modified beam matrix (for mapping from the sky into the SVD basis)
                        dset_bsvd[fi, :nmodes] = beam.reshape(nmodes, self.telescope.num_pol_sky, self.telescope.lmax+1 - mi)

                        # Find the pseudo-inverse of the beam matrix and save to disk.
                        # dset_ibsvd[fi, :, :, :nmodes] = ibeam.reshape(self.telescope.num_pol_sky, self.telescope.lmax + 1, nmodes)

                        # Save out the singular values for each block
                        dset_sig[fi, :nmodes] = sig


                # Write useful attributes.
                fs.attrs['m'] = mi

        mpiutil.barrier()

        if mpiutil.rank0:
            # Make file marker that the m's have been correctly generated:
            open(completed_file, 'a').close()

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()



    def _collect_svd_spectrum(self, regen=False):
        # Gather the SVD spectrum into a single file.

        if os.path.exists(self._svdspectrum_file) and not regen:
            if mpiutil.rank0:
                print
                print '=' * 80
                print "File %s exists. Skipping..." % self._svdspectrum_file
            mpiutil.barrier()
            return


        svd_func = lambda mi: self.beam_singularvalues(mi)

        svdspectrum = kltransform.collect_m_array(self.telescope.mmax+1, svd_func, (self.nfreq, self.svd_len,), np.float64)

        if mpiutil.rank0:
            print
            print '=' * 80
            print 'Create file %s...' % self._svdspectrum_file
            with h5py.File(self._svdspectrum_file, 'w') as f:
                f.create_dataset('singularvalues', data=svdspectrum, compression='lzf')

        mpiutil.barrier()



    def svd_all(self):
        """Collects the full SVD spectrum for all m-modes.

        Reads in from file on disk.

        Returns
        -------
        svarray : np.ndarray[mmax+1, nfreq, svd_len]
            The full set of singular values across all m-modes.
        """

        if not self.svdproj:
            raise Exception('SVD beam singularvalues not generated for self.svdproj == False.')

        with h5py.File(self._svdspectrum_file, 'r') as f:
            svd = f['singularvalues'][:]

        return svd

    #===================================================



    #====== Projection between spaces ==================

    def project_vector_sky_to_telescope(self, mi, vec):
        """Project a vector from the sky into the visibility basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, npol, lmax+1]

        Returns
        -------
        tvec : np.ndarray
            Telescope vector to return.
        """

        vecf = np.zeros((self.nfreq, self.ntel), dtype=np.complex128)

        with h5py.File(self._mfile(mi), 'r') as mfile:

            for fi in range(self.nfreq):
                beamf = mfile['beam_m'][fi][:].reshape((self.ntel, self.nsky(mi)))
                vecf[fi] = np.dot(beamf, vec[fi, :, mi:].reshape(self.nsky(mi)))

        return vecf

    project_vector_forward = project_vector_sky_to_telescope


    def project_vector_telescope_to_sky(self, mi, vec, rank_ratio, lcut):
        """Invert a vector from the telescope space onto the sky. This is the
        map-making process.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [freq, baseline, polarisation]
        rank_ratio : float
            Set :math:`a_{lm}=0` for :math:`\mathbf{Ba=v}` if rank(:math:`\mathbf{B}}`) <= `rank_ratio` * self.nsky(mi). Those alms often cause noisy strips in the final sky map.
        lcut : interger
            Cut threshold of 'l', must be mmax <= lcut <= lmax.

        Returns
        -------
        tvec : np.ndarray
            Sky vector to return.
        """

        beam = self.beam_m(mi)
        npol = beam.shape[-2]
        lside = beam.shape[-1] # lmax+1 - mi
        if lcut is None:
           lcut = lside -1 + mi # lmax
        mmax = self.telescope.mmax
        lcut = max(lcut, mmax) # must l >= m
        lcut1 = lcut + 1
        lcut1 = min(lcut1, lside + mi) # must have lcut <= lmax
        if mpiutil.rank0 and mi == 0:
            print 'Cut l at %d for lmax = %d, mmax = %d.' % (lcut1-1, lside-1+mi, mmax)
        nsky = npol * (lcut1 - mi) # 4 * (lcut + 1 - mi)
        beam = beam[..., :(lcut1 - mi)] # all zero for l < m
        beam = beam.reshape(self.nfreq, self.ntel, nsky)

        vecb = np.zeros((self.nfreq, npol, lside + mi), dtype=np.complex128)
        vec = vec.reshape((self.nfreq, self.ntel))

        for fi in range(self.nfreq):

            if self.noise_weight:
                noisew = self.telescope.noisepower(np.arange(self.telescope.npairs), fi).flatten()**(-0.5)
                noisew = np.concatenate([noisew, noisew])
                beam[fi] = beam[fi] * noisew[:, np.newaxis]
                vec[fi] = vec[fi] * noisew

            # x, resids, rank, s = la.lstsq(np.dot(beam[fi].T.conj(), beam[fi]), np.dot(beam[fi].T.conj(), vec[fi]), cond=1e-6)
            # if rank > rank_ratio * self.nsky(0): # max nsky = nsky(0)
            #     for p in range(npol):
            #         vecb[fi, p, mi:lcut1] = x[p*(lcut1-mi):(p+1)*(lcut1-mi)]
            # else:
            #     print 'Rank <= %.1f for m = %d, fi = %d...' % (rank_ratio*self.nsky(0), mi, fi)

            x = complex_br(np.dot(beam[fi].T.conj(), beam[fi]), np.dot(beam[fi].T.conj(), vec[fi]), n_iter=500, tol=1.0e-6, copy_A=False)
            for p in range(npol):
                vecb[fi, p, mi:lcut1] = x[p*(lcut1-mi):(p+1)*(lcut1-mi)]

        return vecb

    project_vector_backward = project_vector_telescope_to_sky


    def project_vector_telescope_to_sky_full(self, vec, phis, local_freq, maxl=None):
        """Invert a vector from the telescope space onto the sky. This is the
        map-making process."""

        lfreq = len(local_freq)
        nphi = phis.shape[0]
        npol = self.telescope.num_pol_sky
        if maxl is None:
            lside = self.telescope.lmax + 1
        else:
            lside = min(maxl+1, self.telescope.lmax+1)
        nlms = lside**2
        beam = np.zeros((nphi, self.nbase, npol, nlms), dtype=self.beam_l(0).dtype)

        # sum_ephi = np.zeros(2*lside-1, dtype=np.complex128)
        # for mi in range(-lside+1, lside):
        #     sum_ephi[mi] = np.sum(np.exp(1.0J * mi * phis))

        vecb = np.zeros((lfreq, npol, lside, lside), dtype=np.complex128)

        for ind, fi in enumerate(local_freq):
            print 'Map-making for freq: %d of %d...' % (fi, self.nfreq)
            # load the beam from disk for frequency fi
            for li in range(lside):
                tmp = self.beam_l(li, fi)
                shp = tmp.shape
                tmp = np.tile(tmp, (nphi, 1, 1)).reshape((nphi,) + shp)
                beam[..., li**2:(li+1)**2] = tmp
                for pi, phi in enumerate(phis):
                    for mi in range(-li, li+1):
                        beam[pi, ..., li**2+li+mi] *= np.exp(1.0J * mi * phi)
            beam = beam.reshape(nphi*self.nbase, npol*nlms)
            v = vec[ind].T.reshape(-1)

            if self.noise_weight:
                noisew = self.telescope.noisepower(np.arange(self.nbase), fi).flatten()**(-0.5)
                noisew = np.tile(noisew, nphi)
                beam = beam * noisew[:, np.newaxis]
                v = v * noisew

            # print 'Start dot...'
            # lhs = np.dot(beam.T.conj(), beam)
            # print 'Dot done.'
            # # for li in range(lside):
            # #     for mi in range(-li, li+1):
            # #         lhs[li**2+li+mi] *= sum_ephi[mi]
            # print 'Start dot...'
            # rhs = np.dot(beam.T.conj(), vec[ind])
            # print 'Dot done.'

            print 'Start solve...'
            # x, resids, rank, s = la.lstsq(beam, v, cond=1e-2)
            x = complex_br(beam, v, tol=1.0e-6, copy_A=False)
            print 'Solve done.'
            x = x.reshape(npol, nlms)

            # # l1-homotopy
            # h = homotopy.Homotopy(beam, vec[ind])
            # x = h.solve(stop_inc_err=True, warnings=True, verbose=1)
            # x = x.reshape(npol, nlms)

            for li in range(lside):
                for mi in range(0, li+1):
                    vecb[ind, :, li, mi] = 0.5 * (x[:, li**2+li+mi] + (-1)**mi * x[:, li**2+li-mi].conj())
                    # vecb[ind, :, li, mi] = x[:, li**2+li+mi]

        return vecb


    def project_vector_telescope_to_sky_ft(self, vec, phis, local_freq):
        """Invert a vector from the telescope space onto the sky. This is the
        map-making process."""

        lfreq = len(local_freq)
        nphi = phis.shape[0]
        npol = self.telescope.num_pol_sky
        u_max = self.telescope.u_max
        v_max = self.telescope.v_max
        nuvs = (2*u_max+1) * (2*v_max+1)
        beam = np.zeros((nphi, self.nbase, npol, 2*u_max+1, 2*v_max+1), dtype=self.beam_u(0).dtype)

        vecb = np.zeros((lfreq, npol, 2*u_max+1, 2*v_max+1), dtype=np.complex128)

        lon_ext = np.radians(np.sum(self.telescope.lonra)) # radians
        for ind, fi in enumerate(local_freq):
            print 'Map-making for freq: %d of %d...' % (fi, self.nfreq)
            # load the beam from disk for frequency fi
            for ui in range(u_max+1):
                tmp = self.beam_u(ui, fi)
                shp = tmp.shape
                tmp = np.tile(tmp, (nphi, 1, 1, 1)).reshape((nphi,) + shp)
                beam[..., ui, :] = tmp[:, 0]
                if ui != 0:
                    beam[..., -ui, :] = tmp[:, 1]
                for pi, phi in enumerate(phis):
                    # alpha = (2*u_max + 1) * phi / lon_ext
                    alpha =  phi / lon_ext
                    beam[pi, ..., ui, :] *= np.exp(2 * np.pi * 1.0J * ui * alpha)
                    if ui != 0:
                        beam[pi, ..., -ui, :] *= np.exp(-2 * np.pi * 1.0J * ui * alpha)
            beam = beam.reshape(nphi*self.nbase, npol*nuvs)
            v = vec[ind].T.reshape(-1)

            if self.noise_weight:
                noisew = self.telescope.noisepower(np.arange(self.nbase), fi).flatten()**(-0.5)
                noisew = np.tile(noisew, nphi)
                beam = beam * noisew[:, np.newaxis]
                v = v * noisew

            # print 'Start dot...'
            # lhs = np.dot(beam.T.conj(), beam)
            # print 'Dot done.'
            # # for li in range(lside):
            # #     for mi in range(-li, li+1):
            # #         lhs[li**2+li+mi] *= sum_ephi[mi]
            # print 'Start dot...'
            # rhs = np.dot(beam.T.conj(), vec[ind])
            # print 'Dot done.'

            print 'Start solve...'
            x, resids, rank, s = la.lstsq(beam, v, cond=1e-2)
            # x = complex_br(beam, v, tol=1.0e-6, copy_A=False)
            print 'Solve done.'
            x = x.reshape(npol, 2*u_max+1, 2*v_max+1)

            vecb[ind] = x

            # for ui in range(lside):
            #     for mi in range(0, li+1):
            #         vecb[ind, :, li, mi] = 0.5 * (x[:, li**2+li+mi] + (-1)**mi * x[:, li**2+li-mi].conj())
            #         # vecb[ind, :, li, mi] = x[:, li**2+li+mi]

        return vecb


    # def project_vector_backward_dirty(self, mi, vec):

    #     dbeam = self.beam_m(mi).reshape((self.nfreq, self.ntel, self.nsky))
    #     dbeam = dbeam.transpose((0, 2, 1)).conj()

    #     vecb = np.zeros((self.nfreq, self.nsky), dtype=np.complex128)
    #     vec = vec.reshape((self.nfreq, self.ntel))

    #     for fi in range(self.nfreq):
    #         norm = np.dot(dbeam[fi].T.conj(), dbeam[fi]).diagonal()
    #         norm = np.where(norm < 1e-6, 0.0, 1.0 / norm)
    #         #norm = np.dot(dbeam[fi], dbeam[fi].T.conj()).diagonal()
    #         #norm = np.where(np.logical_or(np.abs(norm) < 1e-4,
    #         #np.abs(norm) < np.abs(norm.max()*1e-2)), 0.0, 1.0 / norm)
    #         vecb[fi] = np.dot(dbeam[fi], vec[fi, :].reshape(self.ntel) * norm)

    #     return vecb.reshape((self.nfreq, self.telescope.num_pol_sky,
    #                          self.telescope.lmax + 1))


    def project_matrix_sky_to_telescope(self, mi, mat):
        """Project a covariance matrix from the sky into the visibility basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix packed as [pol, pol, l, freq, freq]

        Returns
        -------
        tmat : np.ndarray
            Covariance in telescope basis.
        """
        npol = self.telescope.num_pol_sky
        lside = self.telescope.lmax + 1

        beam = self.beam_m(mi).reshape((self.nfreq, self.ntel, npol, lside))

        matf = np.zeros((self.nfreq, self.ntel, self.nfreq, self.ntel), dtype=np.complex128)


        # Should it be a +=?
        for pi in range(npol):
            for pj in range(npol):
                for fi in range(self.nfreq):
                    for fj in range(self.nfreq):
                        matf[fi, :, fj, :] += np.dot((beam[fi, :, pi, :] * mat[pi, pj, :, fi, fj]), beam[fj, :, pj, :].T.conj())

        return matf

    project_matrix_forward = project_matrix_sky_to_telescope


    def _svd_num(self, mi):
        ## Calculate the number of SVD modes meeting the cut for each
        ## frequency, return the number and the array bounds

        # Get the array of singular values for each mode
        sv = self.beam_singularvalues(mi)

        # Number of significant sv modes at each frequency
        svnum = (sv > sv.max() * self.svcut).sum(axis=1)

        # Calculate the block bounds within the full matrix
        svbounds = np.cumsum(np.insert(svnum, 0, 0))

        return svnum, svbounds


    def _svd_freq_iter(self, mi):
        num = self._svd_num(mi)[0]
        return [fi for fi in range(self.nfreq) if (num[fi] > 0)]


    def project_matrix_sky_to_svd(self, mi, mat, temponly=False,  pc=None, min_dist=1):
        """Project a covariance matrix from the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix or file containing sky matrix packed as [pol, pol, l, freq, freq].
        Must have pol indices even if `temponly=True`.
        temponly: boolean
            Force projection of temperature (TT) part only (default: False)
        pc :
            Process context to do distributed calculation if not Non.
        min_dist : interger
            Minimum matrix size to do distributed calculation.

        Returns
        -------
        tmat : np.ndarray [nsvd, nsvd] or DistributedMatrix
            Covariance in SVD basis.
        dist : Boolean
            Return distributed matrix if True, else np.ndarray.
        """

        npol = 1 if temponly else self.telescope.num_pol_sky

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)
        nside = svbounds[-1] # ndof(mi)
        gfreqs = self._svd_freq_iter(mi) # global frequency index list
        gflen = len(gfreqs)

        comm = None if pc is None else pc.mpi_comm
        grid_shape = [1, 1] if comm is None else pc.grid_shape
        size = 1 if comm is None else comm.size
        rank = 0 if comm is None else comm.rank # local process rank in comm
        assert size == np.prod(grid_shape), 'Invalid grid_shape'
        grid_pos = (rank / grid_shape[1], rank % grid_shape[1]) # local process position in the process grid

        nr, sr, er = mpiutil.split_m(gflen, grid_shape[0])[:, grid_pos[0]] # local row
        nc, sc, ec = mpiutil.split_m(gflen, grid_shape[1])[:, grid_pos[1]] # local column
        lrfreqs = gfreqs[sr:er] # local row frequency list
        slrfreq, elrfreq = lrfreqs[0], lrfreqs[-1] # start and end local row frequency
        lcfreqs = gfreqs[sc:ec] # local column frequency list
        slcfreq, elcfreq = lcfreqs[0], lcfreqs[-1] # start and end local column frequency

        lrsvnum = [svnum[lfi] for lfi in lrfreqs]
        gr_idx = svbounds[lrfreqs[0]] # start row index in the global matrix corresponding to the local section
        lrnum = np.sum(lrsvnum) # local number of rows of out matrix
        lrbounds = np.cumsum(np.insert(lrsvnum, 0, 0)) # local row bounds
        lcsvnum = [svnum[lfi] for lfi in lcfreqs]
        gc_idx = svbounds[lcfreqs[0]] # start column index in the global matrix corresponding to the local section
        lcnum = np.sum(lcsvnum) # local number of columes of out matrix
        lcbounds = np.cumsum(np.insert(lcsvnum, 0, 0)) # local column bounds

        # read in corresponding section of the beam matrix
        with h5py.File(self._svdfile(mi), 'r') as svdfile:
            rbeam = svdfile['beam_svd'][slrfreq:(elrfreq+1)] # beam section corresponding to the row process
            cbeam = svdfile['beam_svd'][slcfreq:(elcfreq+1)] # beam section corresponding to the column process

        # read in corresponding section of the Cl matrix
        if isinstance(mat, np.ndarray):
            mat = mat[:npol, :npol, :, slrfreq:(elrfreq+1), slcfreq:(elcfreq+1)]
        else:
            with h5py.File(mat, 'r') as cl:
                mat = cl['cv'][:npol, :npol, mi:, slrfreq:(elrfreq+1), slcfreq:(elcfreq+1)]

        # Create the local section of the output matrix
        matf = np.zeros((lrnum, lcnum), dtype=np.complex128, order='C')

        ##------------------------------------------------
        rank0 = True if comm is None or comm.rank == 0 else False
        if rank0:
            print 'Start matf computation...'
            sys.stdout.flush()
        ##----------------------------------------------------

        # each process computes a section of the global matrix
        for pi in range(npol):
            for pj in range(npol):
                for (i, fi) in enumerate(lrfreqs):

                    fibeam = rbeam[fi-slrfreq, :svnum[fi], pi, :] # Beam for this pol, freq, and svcut (i)

                    for (j, fj) in enumerate(lcfreqs):
                        fjbeam = cbeam[fj-slcfreq, :svnum[fj], pj, :] # Beam for this pol, freq, and svcut (j)

                        lmat = mat[pi, pj, :, fi-slrfreq, fj-slcfreq] # Local section of the sky matrix (i.e C_l part)

                        matf[lrbounds[i]:lrbounds[i+1], lcbounds[j]:lcbounds[j+1]] += np.dot(fibeam * lmat, fjbeam.T.conj())

        ##------------------------------------------------
        if rank0:
            print 'matf computation done.'
            sys.stdout.flush()
        ##----------------------------------------------------

        # reduce memory use
        del mat
        del lmat
        del rbeam
        del cbeam
        del fibeam
        del fjbeam

        if comm is None:
            return matf, False
        else:
            gsizes = (nside, nside)
            lsize = (lrnum, lcnum)
            start = (gr_idx, gc_idx)
            lsizes = comm.allgather(lsize)
            starts = comm.allgather(start)

            # for global array less than (min_dist, min_dist), collect all local sections to a global matrix in rank 0
            if nside < min_dist:
                mpi_dtype = mpiutil.typemap(matf.dtype)
                if comm.rank == 0:
                    # create global matrix and subarray datatype view of the global matrix
                    gmatf = np.empty(gsizes, dtype=matf.dtype)
                    subtypes = [mpi_dtype.Create_subarray(gsizes, lsizes[i], starts[i], order=mpiutil.ORDER_C).Commit() for i in range(comm.size)] # default order=ORDER_C
                else:
                    gmatf = None

                # Each process should send its local sections.
                sreq = comm.Isend([matf, mpi_dtype], dest=0, tag=0)

                if comm.rank == 0:
                    # Post each receive
                    reqs = [ comm.Irecv([gmatf, subtypes[sr]], source=sr, tag=0) for sr in range(comm.size) ]

                    # Wait for requests to complete
                    mpiutil.Prequest.Waitall(reqs)

                sreq.Wait()

                return gmatf, False

            # # for global array larger than (min_dist, min_dist), create an distributed matrix
            else:
                # blk_size = (nside - 1) / comm.size + 1
                # blk_shape = (blk_size, blk_size)
                blk_shape = (32, 32) # block size of 16 or 32 is most effective
                matf = np.asfortranarray(matf)
                gmatf = core.DistributedMatrix([nside, nside], dtype=np.complex128, block_shape=blk_shape, context=pc)

                ##----------------------------------------------
                if rank0:
                    print 'Start to copy to gmatf...'
                    sys.stdout.flush()
                ##----------------------------------------------

                # copy the local matrix to the corresponding section of the distributed matrix
                for i in range(comm.size):
                    gmatf = gmatf.np2self(matf, starts[i][0], starts[i][1], block_shape=(20000, 20000), rank=i)


                ##----------------------------------------------
                if rank0:
                    print 'Copy to gmatf done.'
                    sys.stdout.flush()
                ##---------------------------------------------
                return gmatf, True


    def project_matrix_diagonal_telescope_to_svd(self, mi, dmat):
        """Project a diagonal matrix from the telescope basis into the SVD basis.

        This slightly specialised routine is for projecting the noise
        covariance into the SVD space.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix packed as [nfreq, ntel]

        Returns
        -------
        tmat : np.ndarray [nsvd, nsvd]
            Covariance in SVD basis.
        """

        beam = self.beam_ut(mi)

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Create the output matrix
        matf = np.zeros((svbounds[-1], svbounds[-1]), dtype=np.complex128)

        # Should it be a +=?
        for fi in self._svd_freq_iter(mi):

            fbeam = beam[fi, :svnum[fi], :] # Beam matrix for this frequency and cut
            lmat = dmat[fi, :] # Matrix section for this frequency

            matf[svbounds[fi]:svbounds[fi+1], svbounds[fi]:svbounds[fi+1]] = np.dot((fbeam * lmat), fbeam.T.conj())

        return matf


    def project_vector_telescope_to_svd(self, mi, vec):
        """Map a vector from the telescope space into the SVD basis.

        This projection may be lose information about the sky, depending on
        the polarisation filtering.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Telescope data vector packed as [freq, baseline, polarisation]

        Returns
        -------
        svec : np.ndarray[svdnum]
            SVD vector to return.
        """

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Get the SVD beam matrix
        beam = self.beam_ut(mi)

        # Create the output matrix (shape is calculated from input shape)
        vecf = np.zeros((svbounds[-1],) + vec.shape[2:], dtype=np.complex128)

        # Should it be a +=?
        for fi in self._svd_freq_iter(mi):

            fbeam = beam[fi, :svnum[fi], :] # Beam matrix for this frequency and cut
            lvec = vec[fi, :] # Matrix section for this frequency

            vecf[svbounds[fi]:svbounds[fi+1]] = np.dot(fbeam, lvec)

        return vecf


    def project_vector_sky_to_svd(self, mi, vec, temponly=False):
        """Project a vector from the the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, lmax+1]
        temponly: boolean
            Force projection of temperature part only (default: False)

        Returns
        -------
        svec : np.ndarray
            SVD vector to return.
        """
        npol = 1 if temponly else self.telescope.num_pol_sky

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Get the SVD beam matrix
        beam = self.beam_svd(mi)

        # Create the output matrix
        vecf = np.zeros((svbounds[-1],) + vec.shape[3:], dtype=np.complex128)

        for pi in range(npol):
            for fi in self._svd_freq_iter(mi):

                fbeam = beam[fi, :svnum[fi], pi, :] # Beam matrix for this frequency and cut
                lvec = vec[fi, pi] # Matrix section for this frequency

                vecf[svbounds[fi]:svbounds[fi+1]] += np.dot(fbeam, lvec)

        return vecf


    def project_vector_svd_conj(self, mi, vec, temponly=False):
        """Project a vector by conjugate SVD beam matrix.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, lmax+1]
        temponly: boolean
            Force projection of temperature part only (default: False)

        Returns
        -------
        svec : np.ndarray
            SVD vector to return.
        """
        npol = 1 if temponly else self.telescope.num_pol_sky

        svnum, svbounds = self._svd_num(mi)

        # Get the SVD beam matrix
        beam = self.beam_svd(mi)

        # Create the output matrix
        vecf = np.zeros((self.nfreq, self.telescope.num_pol_sky, self.telescope.lmax+1 - mi,) + vec.shape[1:], dtype=np.complex128)

        for pi in range(npol):
            for fi in self._svd_freq_iter(mi):

                fbeam = beam[fi, :svnum[fi], pi, :].T.conj() # Beam matrix for this frequency and cut

                lvec = vec[svbounds[fi]:svbounds[fi+1]] # Matrix section for this frequency

                vecf[fi, pi] += np.dot(fbeam, lvec)

        return vecf


    def project_vector_svd_to_sky(self, mi, vec, rank_ratio, lcut, temponly=False):
        """Project a vector from the the sky into the SVD basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, lmax+1]
        rank_ratio : float
            Set :math:`a_{lm}=0` for :math:`\mathbf{\bar{B}a=\bar{v}}` if rank(:math:`\mathbf{\bar{B}}`) <= `rank_ratio` * (lmax + 1). Those alms often cause noisy strips in the final sky map.
        lcut : interger
            Cut threshold of 'l', must be mmax <= lcut <= lmax.
        temponly: boolean
            Force projection of temperature part only (default: False)

        Returns
        -------
        svec : np.ndarray
            SVD vector to return.
        """
        npol = 1 if temponly else self.telescope.num_pol_sky

        # Number of significant sv modes at each frequency, and the array bounds
        svnum, svbounds = self._svd_num(mi)

        # Get the SVD beam matrix
        beam = self.beam_svd(mi)
        lside = beam.shape[-1] # lmax + 1 -mi
        if lcut is None:
           lcut = lside - 1 + mi
        mmax = self.telescope.mmax
        lcut = max(lcut, mmax) # must l >= m
        lcut1 = lcut + 1
        lcut1 = min(lcut1, lside + mi) # must have lcut <= lmax
        if mpiutil.rank0 and mi == 0:
            print 'Cut l at %d for lmax = %d, mmax = %d.' % (lcut1-1, lside-1+mi, mmax)
        # nsky = npol * (lside - mi) # 4 * (lmax + 1 - mi)
        beam = beam[..., :(lcut1 - mi)] # all zero for l < m

        # Create the output matrix
        vecf = np.zeros((self.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1,) + vec.shape[1:], dtype=np.complex128)

        for pi in range(npol):
            for fi in self._svd_freq_iter(mi):

                lvec = vec[svbounds[fi]:svbounds[fi+1]] # Matrix section for this frequency
                fbeam = beam[fi, :svnum[fi], pi, :]
                x, resids, rank, s = la.lstsq(np.dot(fbeam.T.conj(), fbeam), np.dot(fbeam.T.conj(), lvec), cond=1e-6)
                if rank > rank_ratio * (lside + mi):
                    vecf[fi, pi, mi:lcut1] = x
                else:
                    print ('Rank <= %.1f for m = %d, fi = %d, pol = {%d}...' % (rank_ratio*(lside + mi), mi, fi, pi)).format('T', 'Q', 'U', 'V')

        return vecf



    #===================================================

    #====== Dimensionality of the various spaces =======
    @property
    def nbase(self):
       """The number of unique baselines."""
       return self.telescope.nbase

    @property
    def ntel(self):
        """Degrees of freedom measured by the telescope (per frequency)"""
        return 2 * self.telescope.npairs

    def nsky(self, mi):
        """Degrees of freedom on the sky at each frequency and `m`."""
        # return (self.telescope.lmax + 1) * self.telescope.num_pol_sky
        return (self.telescope.lmax+1 - mi) * self.telescope.num_pol_sky

    @property
    def nfreq(self):
        """Number of frequencies measured."""
        return self.telescope.nfreq

    @property
    def svd_len(self):
        """The size of the SVD output matrices."""
        return min(self.telescope.lmax+1, self.ntel)

    @property
    def ndofmax(self):
        return self.svd_len * self.nfreq

    def ndof(self, mi):
        """The number of degrees of freedom at a given m."""
        return self._svd_num(mi)[1][-1]

    #===================================================



class BeamTransferTempSVD(BeamTransfer):
    """BeamTransfer class that performs the old temperature only SVD.
    """

    def _generate_svdfiles(self, regen=False):
        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._svdfile(mi)) and not regen:
                print ("m index %i. File: %s exists. Skipping..." %
                       (mi, (self._svdfile(mi))))
                continue
            else:
                print 'm index %i. Creating SVD file: %s' % (mi, self._svdfile(mi))

            # Open m beams for reading.
            fm = h5py.File(self._mfile(mi), 'r')

            # Open file to write SVD results into.
            fs = h5py.File(self._svdfile(mi), 'w')

            # Create a chunked dataset for writing the SVD beam matrix into.
            dsize_bsvd = (self.telescope.nfreq, self.svd_len, self.telescope.num_pol_sky, self.telescope.lmax+1)
            csize_bsvd = (1, min(10, self.svd_len), self.telescope.num_pol_sky, self.telescope.lmax+1)
            dset_bsvd = fs.create_dataset('beam_svd', dsize_bsvd, chunks=csize_bsvd, compression='lzf', dtype=np.complex128)

            # Create a chunked dataset for writing the inverse SVD beam matrix into.
            dsize_ibsvd = (self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax+1, self.svd_len)
            csize_ibsvd = (1, self.telescope.num_pol_sky, self.telescope.lmax+1, min(10, self.svd_len))
            dset_ibsvd = fs.create_dataset('invbeam_svd', dsize_ibsvd, chunks=csize_ibsvd, compression='lzf', dtype=np.complex128)

            # Create a chunked dataset for the stokes T U-matrix (left evecs)
            dsize_ut = (self.telescope.nfreq, self.svd_len, self.ntel)
            csize_ut = (1, min(10, self.svd_len), self.ntel)
            dset_ut  = fs.create_dataset('beam_ut', dsize_ut, chunks=csize_ut, compression='lzf', dtype=np.complex128)

            # Create a dataset for the singular values.
            dsize_sig = (self.telescope.nfreq, self.svd_len)
            dset_sig  = fs.create_dataset('singularvalues', dsize_sig, dtype=np.float64)

            ## For each frequency in the m-files read in the block, SVD it,
            ## and construct the new beam matrix, and save.
            for fi in np.arange(self.telescope.nfreq):

                # Read the positive and negative m beams, and combine into one.
                bf = fm['beam_m'][fi][:].reshape(self.ntel, self.telescope.num_pol_sky, self.telescope.lmax + 1)

                noisew = self.telescope.noisepower(np.arange(self.telescope.npairs), fi).flatten()**(-0.5)
                noisew = np.concatenate([noisew, noisew])
                bf = bf * noisew[:, np.newaxis, np.newaxis]

                # Get the T-mode only beam matrix
                bft = bf[:, 0, :]

                # Perform the SVD to find the left evecs
                u, sig, v = svd_gen(bft, full_matrices=False)
                u = u.T.conj() # We only need u^H so just keep that.

                # Save out the evecs (for transforming from the telescope frame into the SVD basis)
                dset_ut[fi] = (u * noisew[np.newaxis, :])

                # Save out the modified beam matrix (for mapping from the sky into the SVD basis)
                bsvd = np.dot(u, bf.reshape(self.ntel, -1))
                dset_bsvd[fi] = bsvd.reshape(self.svd_len, self.telescope.num_pol_sky, self.telescope.lmax + 1)

                # Find the pseudo-inverse of the beam matrix and save to disk.
                dset_ibsvd[fi] = la.pinv(bsvd).reshape(self.telescope.num_pol_sky, self.telescope.lmax + 1, self.svd_len)

                # Save out the singular values for each block
                dset_sig[fi] = sig


            # Write a few useful attributes.
            fs.attrs['baselines'] = self.telescope.baselines
            fs.attrs['m'] = mi
            fs.attrs['frequencies'] = self.telescope.frequencies
            fs.attrs['cylobj'] = self._telescope_pickle

            fs.close()
            fm.close()

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Collect the spectrum into a single file.
        self._collect_svd_spectrum()




class BeamTransferFullSVD(BeamTransfer):
    """BeamTransfer class that performs the old temperature only SVD.
    """

    def _generate_svdfiles(self, regen=False):
        ## Generate all the SVD transfer matrices by simply iterating over all
        ## m, performing the SVD, combining the beams and then write out the
        ## results.

        # For each `m` collect all the `m` sections from each frequency file,
        # and write them into a new `m` file. Use MPI if available.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            if os.path.exists(self._svdfile(mi)) and not regen:
                print ("m index %i. File: %s exists. Skipping..." %
                       (mi, (self._svdfile(mi))))
                continue
            else:
                print 'm index %i. Creating SVD file: %s' % (mi, self._svdfile(mi))

            # Open m beams for reading.
            fm = h5py.File(self._mfile(mi), 'r')

            # Open file to write SVD results into.
            fs = h5py.File(self._svdfile(mi), 'w')

            # Create a chunked dataset for writing the SVD beam matrix into.
            dsize_bsvd = (self.telescope.nfreq, self.svd_len, self.telescope.num_pol_sky, self.telescope.lmax+1)
            csize_bsvd = (1, min(10, self.svd_len), self.telescope.num_pol_sky, self.telescope.lmax+1)
            dset_bsvd = fs.create_dataset('beam_svd', dsize_bsvd, chunks=csize_bsvd, compression='lzf', dtype=np.complex128)

            # Create a chunked dataset for writing the inverse SVD beam matrix into.
            dsize_ibsvd = (self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax+1, self.svd_len)
            csize_ibsvd = (1, self.telescope.num_pol_sky, self.telescope.lmax+1, min(10, self.svd_len))
            dset_ibsvd = fs.create_dataset('invbeam_svd', dsize_ibsvd, chunks=csize_ibsvd, compression='lzf', dtype=np.complex128)

            # Create a chunked dataset for the stokes T U-matrix (left evecs)
            dsize_ut = (self.telescope.nfreq, self.svd_len, self.ntel)
            csize_ut = (1, min(10, self.svd_len), self.ntel)
            dset_ut  = fs.create_dataset('beam_ut', dsize_ut, chunks=csize_ut, compression='lzf', dtype=np.complex128)

            # Create a dataset for the singular values.
            dsize_sig = (self.telescope.nfreq, self.svd_len)
            dset_sig  = fs.create_dataset('singularvalues', dsize_sig, dtype=np.float64)

            ## For each frequency in the m-files read in the block, SVD it,
            ## and construct the new beam matrix, and save.
            for fi in np.arange(self.telescope.nfreq):

                # Read the positive and negative m beams, and combine into one.
                bf = fm['beam_m'][fi][:].reshape(self.ntel, self.telescope.num_pol_sky, self.telescope.lmax + 1)

                noisew = self.telescope.noisepower(np.arange(self.telescope.npairs), fi).flatten()**(-0.5)
                noisew = np.concatenate([noisew, noisew])
                bf = bf * noisew[:, np.newaxis, np.newaxis]

                bf = bf.reshape(self.ntel, -1)

                # Perform the SVD to find the left evecs
                u, sig, v = svd_gen(bf, full_matrices=False)
                u = u.T.conj() # We only need u^H so just keep that.

                # Save out the evecs (for transforming from the telescope frame into the SVD basis)
                dset_ut[fi] = (u * noisew[np.newaxis, :])

                # Save out the modified beam matrix (for mapping from the sky into the SVD basis)
                bsvd = np.dot(u, bf)
                dset_bsvd[fi] = bsvd.reshape(self.svd_len, self.telescope.num_pol_sky, self.telescope.lmax + 1)

                # Find the pseudo-inverse of the beam matrix and save to disk.
                dset_ibsvd[fi] = la.pinv(bsvd).reshape(self.telescope.num_pol_sky, self.telescope.lmax + 1, self.svd_len)

                # Save out the singular values for each block
                dset_sig[fi] = sig


            # Write a few useful attributes.
            fs.attrs['baselines'] = self.telescope.baselines
            fs.attrs['m'] = mi
            fs.attrs['frequencies'] = self.telescope.frequencies
            fs.attrs['cylobj'] = self._telescope_pickle

            fs.close()
            fm.close()

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()

        # Collect the spectrum into a single file.
        self._collect_svd_spectrum()

    @property
    def svd_len(self):
        """The size of the SVD output matrices."""
        return min((self.telescope.lmax+1) * self.telescope.num_pol_sky, self.ntel)



class BeamTransferNoSVD(BeamTransfer):

    svdproj = False
    svcut = 0.0

    def __init__(self, directory, telescope=None, svdproj=True):

        self.directory = directory
        self.telescope = telescope
        self.svdproj = False

        # Create directory if required
        if mpiutil.rank0 and not os.path.exists(directory):
            os.makedirs(directory)

        mpiutil.barrier()

        if self.telescope == None and mpiutil.rank0:
            print "Attempting to read telescope from disk..."

            try:
                f = open(self._picklefile, 'r')
                self.telescope = pickle.load(f)
            except IOError, UnpicklingError:
                raise Exception("Could not load Telescope object from disk.")


    def project_matrix_sky_to_svd(self, mi, mat, *args, **kwargs):
        return self.project_matrix_sky_to_telescope(mi, mat).reshape(self.ndof(mi), self.ndof(mi))


    def project_vector_sky_to_svd(self, mi, vec, *args, **kwargs):
        return self.project_vector_sky_to_telescope(mi, vec)


    def project_matrix_diagonal_telescope_to_svd(self, mi, dmat, *args, **kwargs):
        return np.diag(dmat.flatten())

    def project_vector_telescope_to_svd(self, mi, vec, *args, **kwargs):
        return vec.reshape(-1)

    def project_vector_svd_to_sky(self, mi, vec, rank_ratio, lcut, temponly=False):
        return self.project_vector_telescope_to_sky(mi, vec, rank_ratio, lcut)


    def beam_svd(self, mi, *args, **kwargs):
        return self.beam_m(mi)



    def ndof(self, mi, *args, **kwargs):

        return self.ntel * self.nfreq

    @property
    def ndofmax(self):
        return self.ntel * self.nfreq
