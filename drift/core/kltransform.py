import time
import os
import re

import numpy as np
import scipy.linalg as la
import h5py

from cora.util import hputil

from drift.util import mpiutil, util, config
from drift.util import typeutil
from drift.core import skymodel

from scalapy import core
import scalapy.routines as rt
from drift.util import scalapyutil as su


def collect_m_arrays(mlist, func, shapes, dtype):

    data = [ (mi, func(mi)) for mi in mpiutil.partition_list_mpi(mlist) ]

    mpiutil.barrier()

    if mpiutil.rank0 and mpiutil.size == 1:
        p_all = [data]
    else:
        p_all = mpiutil.world.gather(data, root=0)

    mpiutil.barrier() # Not sure if this barrier really does anything,
                      # but hoping to stop collect breaking

    marrays = None
    if mpiutil.rank0:
        marrays = [np.zeros((len(mlist),) + shape, dtype=dtype) for shape in shapes]

        for p_process in p_all:

            for mi, result in p_process:

                for si in range(len(shapes)):
                    if result[si] is not None:
                        marrays[si][mi] = result[si]

    mpiutil.barrier()

    return marrays


def collect_m_array(mlist, func, shape, dtype):

    res = collect_m_arrays(mlist, lambda mi: [func(mi)], [shape], dtype)

    return res[0] if mpiutil.rank0 else None




def eigh_gen(A, B):
    """Solve the generalised eigenvalue problem. :math:`\mathbf{A} \mathbf{v} =
    \lambda \mathbf{B} \mathbf{v}`

    This routine will attempt to correct for when `B` is not positive definite
    (usually due to numerical precision), by adding a constant diagonal to make
    all of its eigenvalues positive.

    Parameters
    ----------
    A, B : np.ndarray
        Matrices to operate on.

    Returns
    -------
    evals : np.ndarray
        Eigenvalues of the problem.
    evecs : np.ndarray
        2D array of eigenvectors (packed column by column).
    add_const : scalar
        The constant added on the diagonal to regularise.
    """
    add_const = 0.0

    if (A == 0).all():
        evals, evecs = np.zeros(A.shape[0], dtype=A.real.dtype), np.identity(A.shape[0], dtype=A.dtype)

    else:

        try:
            evals, evecs = la.eigh(A, B, overwrite_a=True, overwrite_b=True)
        except la.LinAlgError as e:
            print "Error occured in eigenvalue solve."
            # Get error number
            mo = re.search('order (\\d+)', e.message)

            # If exception unrecognised then re-raise.
            if mo is None:
                raise e

            errno = mo.group(1)

            if errno < (A.shape[0]+1):

                print "Matrix probably not positive definite due to numerical issues. \
                Trying to add a constant diagonal...."

                evb = la.eigvalsh(B)
                add_const = 1e-15 * evb[-1] - 2.0 * evb[0] + 1e-60

                B[np.diag_indices(B.shape[0])] += add_const
                evals, evecs = la.eigh(A, B, overwrite_a=True, overwrite_b=True)

            else:
                print "Strange convergence issue. Trying non divide and conquer routine."
                evals, evecs = la.eigh(A, B, overwrite_a=True, overwrite_b=True, turbo=False)

    return evals, evecs, add_const


def inv_gen(A):
    """Find the inverse of A.

    If a standard matrix inverse has issues try using the pseudo-inverse.

    Parameters
    ----------
    A : np.ndarray
        Matrix to invert.

    Returns
    -------
    inv : np.ndarray
    """
    try:
        inv = la.inv(A)
    except la.LinAlgError:
        inv = la.pinv(A)

    return inv



class KLTransform(config.Reader):
    """Perform KL transform.

    Attributes
    ----------
    subset : boolean
        If True, throw away modes below a S/N `threshold`.
    threshold : scalar
        S/N threshold to cut modes at.
    inverse : boolean
        If True construct and cache inverse transformation.
    use_thermal, use_foregrounds : boolean
        Whether to use instrumental noise/foregrounds (default: both True)
    _foreground_regulariser : scalar
        The regularisation constant for the foregrounds. Adds in a diagonal of
        size reg * cf.max(). Default is 2e-15
    """

    subset = config.Property(proptype=bool, default=True, key='subset')
    inverse = config.Property(proptype=bool, default=False, key='inverse')

    threshold = config.Property(proptype=typeutil.nonnegative_float, default=0.1, key='threshold')

    _foreground_regulariser = config.Property(proptype=typeutil.nonnegative_float, default=1e-14, key='regulariser')

    use_thermal = config.Property(proptype=bool, default=True)
    use_foregrounds = config.Property(proptype=bool, default=True)
    use_polarised = config.Property(proptype=bool, default=True)

    pol_length = config.Property(proptype=typeutil.none_or_positive_float, default=None)

    klname = None

    # _cvfg = None
    # _cvsg = None

    distribute = True # do distribute calculation if True
    # spcomm = mpiutil.world # splited communicator
    grid_shape = [2, 3] # process grid shape, take effect only if distribute == True
    # pc = core.ProcessContext([2, 3], comm=spcomm) # process context
    # pc = None
    min_dist = 100 # minimum matrix size to do the distributed calculation, take effect only if distribute == True

    @property
    def _cvdir(self):
        return self.beamtransfer.directory + '/covariance/'

    @property
    def _cvfg(self):
        return self._cvdir + 'fg_covariance.hdf5'

    @property
    def _cvsg(self):
        return self._cvdir + 'sg_covariance.hdf5'

    @property
    def _evdir(self):
        return self.beamtransfer.directory + '/' + self.klname + '/'

    def _evfile(self, mi):
        # Pattern to form the `m` ordered file.
        pat = self._evdir + 'ev_m_%s.hdf5' % util.natpattern(self.telescope.mmax)
        return pat % abs(mi)

    @property
    def _all_evfile(self):
        return self._evdir + 'evals.hdf5'

    def __init__(self, bt, klname):
        self.beamtransfer = bt
        self.telescope = self.beamtransfer.telescope
        self.klname = klname

        mpiutil.barrier()


    def foreground(self):
        """Compute the foreground covariance matrix (on the sky).

        Returns
        -------
        cv_fg : np.ndarray[pol2, pol1, l, freq1, freq2]
        """

        npol = self.telescope.num_pol_sky

        if npol != 1 and npol != 3 and npol != 4:
            raise Exception("Can only handle unpolarised only (num_pol_sky \
                             = 1), or I, Q and U (num_pol_sky = 3).")

        # If not polarised then zero out the polarised components of the array
        if self.use_polarised:
            return skymodel.foreground_model(self.telescope.lmax,
                                             self.telescope.frequencies,
                                             npol, pol_length=self.pol_length)
        else:
            return skymodel.foreground_model(self.telescope.lmax,
                                             self.telescope.frequencies,
                                             npol, pol_frac=0.0)


    def signal(self):
        """Compute the signal covariance matrix (on the sky).

        Returns
        -------
        cv_fg : np.ndarray[pol2, pol1, l, freq1, freq2]
        """

        npol = self.telescope.num_pol_sky

        if npol != 1 and npol != 3 and npol != 4:
            raise Exception("Can only handle unpolarised only (num_pol_sky \
                            = 1), or I, Q and U (num_pol_sky = 3).")

        return skymodel.im21cm_model(self.telescope.lmax,
                                           self.telescope.frequencies, npol)


    def _generate_cvfg(self, regen=False):
        if os.path.exists(self._cvfg) and not regen:
            print 'File %s exists. Skipping...' % self._cvfg
            return

        with h5py.File(self._cvfg, 'w') as f:
            f.create_dataset('cv', data=self.foreground())


    def _generate_cvsg(self, regen=False):
        if os.path.exists(self._cvsg) and not regen:
            print 'File %s exists. Skipping...' % self._cvsg
            return

        with h5py.File(self._cvsg, 'w') as f:
            f.create_dataset('cv', data=self.signal())


    def cvfg_m(self, mi):
        with h5py.File(self._cvfg, 'r') as f:
            return f['cv'][:, :, mi:, :, :] # only l >= m


    def cvsg_m(self, mi):
        assert os.path.exists(self._cvfg)
        with h5py.File(self._cvsg, 'r') as f:
            return f['cv'][:, :, mi:, :, :] # only l >= m


    def sn_covariance(self, mi, comm=None):
        """Compute the signal and noise covariances (on the telescope).

        The signal is formed from the 21cm signal, whereas the noise includes
        both foregrounds and instrumental noise. This is for a single m-mode.

        Parameters
        ----------
        mi : integer
            The m-mode to calculate at.

        Returns
        -------
        s, n : np.ndarray[nfreq, ntel, nfreq, ntel]
            Signal and noice covariance matrices.
        """

        rank0 = True if comm is None or comm.Get_rank() == 0 else False
        rank1 = True if comm is None or comm.Get_rank() == 1 else False
        if not (self.use_foregrounds or self.use_thermal):
            raise Exception("Either `use_thermal` or `use_foregrounds`, or both must be True.")

        # Project the signal and foregrounds from the sky onto the telescope.

        # print comm.Get_size()
        # comm.Barrier()
        # core.initmpi([2, 2], block_shape=[10, 10], comm=comm)

        nside = self.beamtransfer.ndof(mi)
        comm_size = 1 if comm is None else comm.Get_size()
        # blk_size = (nside - 1) / comm_size + 1


        dist = False # use distribute computation if True
        cvb_s = None
        cvb_n = None

        # comm.Barrier()

        if rank0:
            print 'Start signal covariance projection for m = %d...' % mi
            cvb_s = self.beamtransfer.project_matrix_sky_to_svd(mi, self.cvsg_m(mi))
            print 'Signal covariance projection for m = %d done.' % mi
            if comm is not None and nside >= self.min_dist:
                cvb_s = np.asfortranarray(cvb_s)
                dist = True
            #     comm.bcast(dist, root=0)
            # else:
            #     comm.Bcast(cvb_s, root=0)
            # assert np.allclose(cvb_s, cvb_s.T.conj())
            # print 'cvb_s.shape = ', cvb_s.shape
        # selected_rank = 1 if comm_size >=2 else 0
        # if comm is None or comm.Get_rank() == selected_rank:
        if rank1:
            print 'Start foreground covariance projection for m = %d...' % mi
            if self.use_foregrounds:
                cvb_n = self.beamtransfer.project_matrix_sky_to_svd(mi, self.cvfg_m(mi))
            else:
                cvb_n = np.zeros_like(cvb_s)
            print 'Foreground covariance projection for m = %d done.' % mi

            # Add in a small diagonal to regularise the noise matrix.
            cnr = cvb_n.reshape((self.beamtransfer.ndof(mi), -1))
            cnr[np.diag_indices_from(cnr)] += self._foreground_regulariser * cnr.max()

            # Even if noise=False, we still want a very small amount of
            # noise, so we multiply by a constant to turn Tsys -> 1 mK.
            nc = 1.0
            if not self.use_thermal:
                nc =  (1e-3 / self.telescope.tsys_flat)**2

            # Construct diagonal noise power in telescope basis
            bl = np.arange(self.telescope.npairs)
            bl = np.concatenate((bl, bl))
            npower = nc * self.telescope.noisepower(bl[np.newaxis, :], np.arange(self.telescope.nfreq)[:, np.newaxis]).reshape(self.telescope.nfreq, self.beamtransfer.ntel)

            # Project into SVD basis and add into noise matrix
            cvb_n += self.beamtransfer.project_matrix_diagonal_telescope_to_svd(mi, npower)
            if comm is not None and nside >= self.min_dist:
                cvb_n = np.asfortranarray(cvb_n)
                dist = True
            #     comm.bcast(dist, root=selected_rank)
            # else:
            #     comm.Bcast(cvb_n, root=selected_rank)
            # cvb_n = np.asfortranarray(cvb_n)
            # assert np.allclose(cvb_n, cvb_n.T.conj())
            # print 'cvb_n.shape = ', cvb_n.shape

        if comm is not None:
            comm.Barrier()
            dist = comm.bcast(dist, root=0)

        # print 'Process %d dist = ' % comm.Get_rank(), dist

        if dist == True:
            # get block size according to nside
            blk_size = (nside - 1) / comm_size + 1
            pc = core.ProcessContext(self.grid_shape, comm=comm) # process context
            dcvb_s = core.DistributedMatrix.from_global_array(cvb_s, rank=0, block_shape=[blk_size, blk_size], context=pc)
            dcvb_n = core.DistributedMatrix.from_global_array(cvb_n, rank=1, block_shape=[blk_size, blk_size], context=pc)
            return dcvb_s, dcvb_n, True
        else:
            # nside = self.beamtransfer.ndof(mi)
            if cvb_s is None:
                cvb_s = np.empty((nside, nside), dtype=np.complex128)
            if cvb_n is None:
                cvb_n = np.empty((nside, nside), dtype=np.complex128)
            # comm.Bcast([cvb_s, mpiutil.typemap(cvb_s.dtype)], [cvb_s, mpiutil.typemap(cvb_s.dtype)], root=0) # more effective to use Bcast
            # comm.Bcast([cvb_n, mpiutil.typemap(cvb_n.dtype)], [cvb_n, mpiutil.typemap(cvb_n.dtype)], root=selected_rank)
            if comm is not None:
                comm.Bcast(cvb_s, root=0) # more effective to use Bcast
                comm.Bcast(cvb_n, root=1)
            return cvb_s, cvb_n, False

        # if comm.Get_rank() == 0:
        #     print 'Return projected distribute covariance matrices for m = %d.' % mi
        # # return cvb_s, cvb_n
        # return dcvb_s, dcvb_n


    def _transform_m(self, mi, comm=None):
        """Perform the KL-transform for a single m.

        Parameters
        ----------
        mi : integer
            The m-mode to calculate for.

        Returns
        -------
        evals, evecs : np.ndarray
            The KL-modes. The evals correspond to the diagonal of the
            covariances in the new basis, and the evecs define the basis.
        """

        rank0 = True if comm is None or comm.Get_rank() == 0 else False
        if rank0:
             print "Solving for Eigenvalues...."

        # Fetch the covariance matrices to diagonalise
        st = time.time()
        nside = self.beamtransfer.ndof(mi)

        # Ensure that number of SVD degrees of freedom is non-zero before proceeding
        if nside == 0:
            return np.array([]), np.array([[]]), np.array([[]]), { 'ac' : 0.0 }

        # cvb_sr, cvb_nr = [cv.reshape(nside, nside) for cv in self.sn_covariance(mi)]
        cvb_sr, cvb_nr, dist = self.sn_covariance(mi, comm)
        et = time.time()
        if rank0:
            print "Time =", (et-st)

        # Perform the generalised eigenvalue problem to get the KL-modes.
        st = time.time()
        if dist:
            evals, evecs = su.eigh_gen(cvb_sr, cvb_nr)
            evecs = evecs.to_global_array() # no need Hermitian transpose
            # evals, evecs = rt.eigh(cvb_sr, cvb_nr)
            # evecs = evecs.to_global_array()
            # evecs = evecs.T.conj()
            ac = 0.0
        else:
            # print 'Process %d: ' % comm.Get_rank(), cvb_sr, cvb_nr
            evals, evecs, ac = eigh_gen(cvb_sr, cvb_nr)
            evecs = evecs.T.conj() # need Hermitian transpose
        et=time.time()
        if rank0:
            print "Time =", (et-st)

        # evecs = evecs.to_global_array(rank=mpiutil.rank)
        # evecs = evecs.T.conj()
        # evecs = evecs.to_global_array() # ??????
        # evecs = evecs.T.conj()

        # Generate inverse if required
        inv = None
        if self.inverse:
            inv = inv_gen(evecs).T

        # Construct dictionary of extra parameters to return
        evextra = {'ac' : ac}

        return evals, evecs, inv, evextra



    def _transform_save_m(self, mi, comm=None):
        """Save the KL-modes for a given m.

        Perform the transform and cache the results for later use.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.
        comm : mpi4py.MPI.Comm, optional
            The MPI communicator to create a BLACS context for. If comm=None,
            create no BLACS context, i.e., do the usual single process calculation.

        Results
        -------
        evals, evecs : np.ndarray
            See `transfom_m` for details.
        """

        # Perform the KL-transform
        rank0 = True if comm is None or comm.Get_rank() == 0 else False
        if rank0:
            print "Constructing signal and noise covariances for m = %d ..." % mi
        evals, evecs, inv, evextra = self._transform_m(mi, comm)

        ## Write out Eigenvals and Vectors

        # Create file and set some metadata
        if rank0:
            print "Creating file %s ...." % (self._evfile(mi))
            with h5py.File(self._evfile(mi), 'w') as f:
                f.attrs['m'] = mi
                f.attrs['SUBSET'] = self.subset

                ## If modes have been already truncated (e.g. DoubleKL) then pad out
                ## with zeros at the lower end.
                nside = self.beamtransfer.ndof(mi)
                evalsf = np.zeros(nside, dtype=np.float64)
                if evals.size != 0:
                    evalsf[(-evals.size):] = evals
                f.create_dataset('evals_full', data=evalsf, compression='lzf')

                # Discard eigenmodes with S/N below threshold if requested.
                if self.subset:
                    i_ev = np.searchsorted(evals, self.threshold)

                    evals = evals[i_ev:]
                    evecs = evecs[i_ev:]
                    print "Modes with S/N > %f: %i of %i" % (self.threshold, evals.size, evalsf.size)

                # Write out potentially reduced eigen spectrum.
                f.create_dataset('evals', data=evals, compression='lzf')
                f.create_dataset('evecs', data=evecs, compression='lzf')
                f.attrs['num_modes'] = evals.size

                if self.inverse:
                    if self.subset:
                        inv = inv[i_ev:]

                    f.create_dataset('evinv', data=inv, compression='lzf')

                # Call hook which allows derived classes to save special information
                # into the EV file.
                self._ev_save_hook(f, evextra)


    def _transform_save(self, regen=False):
        # Perform the KL-transform and save the KL-mode
        completed_file = self._evdir + 'COMPLETED_EV'
        if os.path.exists(completed_file) and not regen:
            if mpiutil.rank0:
                print
                print '=' * 80
                print "******* %s-files already generated ********" % self.klname
            mpiutil.barrier()
            return

        if mpiutil.rank0:
            st = time.time()
            print
            print '=' * 80
            print "======== Starting covariance matrices calculation ========"

        # mpiutil.barrier()

        # make covariance matrices directory
        try:
            if not os.path.exists(self._cvdir):
                os.makedirs(self._cvdir)
        except OSError:
            pass

        # generate covariance matrices
        # if mpiutil.size >= 2:
        #     if mpiutil.rank0:
        #         self._generate_cvfg(regen)
        #     elif mpiutil.rank == 1:
        #         self._generate_cvsg(regen)
        # else:
        #     self._generate_cvfg(regen)
        #     self._generate_cvsg(regen)

        # generate covariance matrices
        if mpiutil.rank0:
            self._generate_cvsg(regen)
        selected_rank = 1 if mpiutil.size >= 2 else 0
        if mpiutil.rank == selected_rank:
            self._generate_cvfg(regen)

        mpiutil.barrier()

        if mpiutil.rank0:
            et = time.time()
            print "======== Ending covariance matrices calculation (time=%f) ========" % (et - st)

        # mpiutil.barrier()

##------------------------------------------------------------------

        if mpiutil.rank0:
            st = time.time()
            print
            print '=' * 80
            print "======== Starting %s calculation ========" % self.klname

        mpiutil.barrier()

        # Iterate list over MPI processes.
        # grpsize = 4 # group size
        # numgrp = (mpiutil.size - 1) / grpsize + 1 # number of groups
        # # color = mpiutil.rank % grpsize
        # color = mpiutil.rank / grpsize
        # print 'color = %d' % color
        # # split to `numgrp` independent communicators
        # # spcomm = mpiutil.world.Split(color=color, key=mpiutil.rank)
        # spcomm = mpiutil.world.Split(color=0, key=mpiutil.rank)
        # print 'spcomm = ', spcomm
        # print 'world =', mpiutil.world
        # print 'world.Dup =', mpiutil.world.Dup()
        # # core.initmpi([2, 2], block_shape=[10, 10])
        # core.initmpi([2, 2], block_shape=[10, 10], comm=spcomm)
        # # core.initmpi([2, 2], block_shape=[10, 10], comm=mpiutil.world.Dup())
        # # core.initmpi([2, 2], block_shape=[10, 10], comm=mpiutil.world)

        # core.initmpi([2, 3], block_shape=[2, 2])


        if self.grid_shape == [1, 1]:
            self.distribute = False

        if self.distribute == False:
            spcomm = None
            num_grp = mpiutil.size
            color = mpiutil.rank
        else:
            # initialize the distribute calculation communicators
            core.ProcessContext([1, mpiutil.size], comm=None) # process context

            grid_size = np.prod(self.grid_shape)
            if mpiutil.size % grid_size == 0:
                num_grp = mpiutil.size / grid_size  # number of groups
            else:
                raise Exception('Can not divide all processes evenly to each groups')
            color = mpiutil.rank % num_grp
            # orin_grp = mpiutil.world.Get_group()
            # if color == 0:
            #     new_grp = orin_grp.Incl([0, 2, 4, 6, 8, 10])
            # else:
            #     new_grp = orin_grp.Incl([1, 3, 5, 7, 9, 11])
            # spcomm = mpiutil.world.Create(new_grp)
            spcomm = mpiutil.world.Split(color, key=mpiutil.rank)

        # for mi in mpiutil.mpirange(self.telescope.mmax+1):
        # for mi in range(self.telescope.mmax+1):
        for mi in mpiutil.partition_list_alternate(range(self.telescope.mmax+1), color, num_grp):
            # Make directory for kl transform
            try:
                if not os.path.exists(self._evdir):
                    os.makedirs(self._evdir)
            except OSError:
                pass

            if os.path.exists(self._evfile(mi)) and not regen:
                if mpiutil.rank0:
                # if spcomm.Get_rank() == 0:
                    print "File %s exists. Skipping..." % self._evfile(mi)
                continue

            self._transform_save_m(mi, spcomm)
            # self._transform_save_m(mi, self.spcomm)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier() # global synchronization
        # spcomm.Barrier() # synchronize within individual communicators

        if mpiutil.rank0:
        # if spcomm.Get_rank() == 0:
            # Make file marker that the m's have been correctly generated:
            open(completed_file, 'a').close()

        # If we're part of an MPI run, synchronise here.
        # mpiutil.barrier()

        if mpiutil.rank0:
        # if spcomm.Get_rank() == 0:
            et = time.time()
            print "======== Ending %s calculation (time=%f) ========" % (self.klname, (et - st))

        mpiutil.barrier() # global synchronization


    def _ev_save_hook(self, f, evextra):

        ac = evextra['ac']

        # If we had to regularise because the noise spectrum is numerically ill
        # conditioned, write out the constant we added to the diagonal (see
        # eigh_gen).
        if ac != 0.0:
            f.attrs['add_const'] = ac
            f.attrs['FLAGS'] = 'NotPositiveDefinite'
        else:
            f.attrs['FLAGS'] = 'Normal'


    def evals_all(self):
        """Collects the full eigenvalue spectrum for all m-modes.

        Reads in from files on disk.

        Returns
        -------
        evarray : np.ndarray
            The full set of eigenvalues across all m-modes.
        """

        with h5py.File(self._all_evfile, 'r') as f:
            ev = f['evals'][:]

        return ev


    def _collect(self, regen=False):

        if os.path.exists(self._all_evfile) and not regen:
            if mpiutil.rank0:
                print
                print '=' * 80
                print "File %s exists. Skipping..." % self._all_evfile
            mpiutil.barrier()
            return

        if mpiutil.rank0:
            print
            print '=' * 80
            print "Creating eigenvalues file for %s..." % self.klname

        def evfunc(mi):
            evf = np.zeros(self.beamtransfer.ndofmax)

            # ensure that data files has already been saved to disk at the time of reading (file I/O is much slower than CPU)
            while True:
                try:
                    with h5py.File(self._evfile(mi), 'r') as f:
                        if f['evals_full'].shape[0] > 0:
                            ev = f['evals_full'][:]
                            evf[-ev.size:] = ev

                    break
                except IOError:
                    pass

            return evf

        mlist = range(self.telescope.mmax+1)
        shape = (self.beamtransfer.ndofmax, )
        evarray = collect_m_array(mlist, evfunc, shape, np.float64)

        if mpiutil.rank0:
            with h5py.File(self._all_evfile, 'w') as f:
                f.create_dataset('evals', data=evarray, compression='lzf')

        mpiutil.barrier()


    def generate(self, regen=False):
        """Perform the KL-transform for all m-modes and save the result.

        Uses MPI to distribute the work (if available).

        Parameters
        ----------
        mlist : array_like, optional
            Set of m's to calculate KL-modes for By default do all m-modes.
        """

        # Perform the KL-transform for all m-modes and save the result.
        self._transform_save(regen)

        # mpiutil.barrier()

        # Collect together the eigenvalues
        self._collect(regen)



    olddatafile = False

    @util.cache_last
    def modes_m(self, mi, threshold=None):
        """Fetch the KL-modes for a particular m.

        This attempts to read in the results from disk, if available and if not
        will create them.

        Also, it will cache the previous m-mode in memory, so as to avoid disk
        access in many cases. However *this* is not sensitive to changes in the
        threshold, be careful.

        Parameters
        ----------
        mi : integer
            m to fetch KL-modes for.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        evals, evecs : np.ndarray
            KL-modes with S/N greater than some threshold. Both evals and evecs
            are potentially `None`, if there are no modes either in the file, or
            satisfying S/N > threshold.
        """

        # If modes not already saved to disk, create file.
        completed_file = self._evdir + 'COMPLETED_EV'
        if not os.path.exists(completed_file):
            self._transform_save()

        with h5py.File(self._evfile(mi), 'r') as f:
            # If no modes are in the file, return None, None
            if f['evals'].shape[0] == 0:
                modes = None, None
            else:
                # Find modes satisfying threshold (if required).
                evals = f['evals'][:]
                startind = np.searchsorted(evals, threshold) if threshold is not None else 0

                if startind == evals.size:
                    modes = None, None
                else:
                    modes = ( evals[startind:], f['evecs'][startind:] )

                    # If old data file perform complex conjugate
                    modes = modes if not self.olddatafile else ( modes[0], modes[1].conj() )

        return modes


    @util.cache_last
    def evals_m(self, mi, threshold=None):
        """Fetch the KL-modes for a particular m.

        This attempts to read in the results from disk, if available and if not
        will create them.

        Also, it will cache the previous m-mode in memory, so as to avoid disk
        access in many cases. However *this* is not sensitive to changes in the
        threshold, be careful.

        Parameters
        ----------
        mi : integer
            m to fetch KL-modes for.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        evals : np.ndarray
            KL-modes with S/N greater than some threshold. Both evals and evecs
            are potentially `None`, if there are no modes either in the file, or
            satisfying S/N > threshold.
        """

        return self.modes_m(mi, threshold)[0]


    @util.cache_last
    def invmodes_m(self, mi, threshold=None):
        """Get the inverse modes.

        If the true inverse has been cached, return the modes for the current
        `threshold`. Otherwise generate the Moore-Penrose pseudo-inverse.

        Parameters
        ----------
        mi : integer
            m-mode to generate for.
        threshold : scalar
            S/N threshold to use.

        Returns
        -------
        invmodes : np.ndarray
        """

        evals = self.evals_m(mi, threshold)

        with h5py.File(self._evfile(mi), 'r') as f:
            if 'evinv' in f:
                inv = f['evinv'][:]

                if threshold != None:
                    nevals = evals.size
                    inv = inv[(-nevals):]

                return inv.T

            else:
                print "Inverse not cached, generating pseudo-inverse."
                return la.pinv(self.modes_m(mi, threshold)[1])


    @util.cache_last
    def skymodes_m(self, mi, threshold=None):
        """Find the representation of the KL-modes on the sky.

        Use the beamtransfers to rotate the SN-modes onto the sky. This routine
        is based on `modes_m`, as such the same caching and caveats apply.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        skymodes : np.ndarray
            The modes as found in a_{lm}(\nu) space. Note this routine does not
            return the evals.

        See Also
        --------
        `modes_m`
        """

        # Fetch the modes in the telescope basis.
        evals, evecs = self.modes_m(mi, threshold=threshold)

        if evals is None:
            raise Exception("Don't seem to be any evals to use.")

        bt = self.beamtransfer

        ## Rotate onto the sky basis. Slightly complex as need to do
        ## frequency-by-frequency
        beam = self.beamtransfer.beam_m(mi).reshape((bt.nfreq, bt.ntel, bt.nsky))
        evecs = evecs.reshape((-1, bt.nfreq, bt.ntel))

        evsky = np.zeros((evecs.shape[0], bt.nfreq, bt.nsky), dtype=np.complex128)

        for fi in range(bt.nfreq):
            evsky[:, fi, :] = np.dot(evecs[:, fi, :], beam[fi])

        return evsky




    def project_vector_svd_to_kl(self, mi, vec, threshold=None):
        """Project a telescope data vector into the eigenbasis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Telescope data vector.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projvector : np.ndarray
            The vector projected into the eigenbasis.
        """
        evals, evecs = self.modes_m(mi, threshold)

        if evals is None:
            return np.zeros((0,), dtype=np.complex128)

        if vec.shape[0] != evecs.shape[1]:
            raise Exception("Vectors are incompatible.")

        return np.dot(evecs, vec)


    def project_vector_kl_to_svd(self, mi, vec, threshold=None):
        """Project a vector in the Eigenbasis back into the telescope space.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Eigenbasis data vector.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projvector : np.ndarray
            The vector projected into the eigenbasis.
        """
        evals, evecs = self.modes_m(mi, threshold)

        if evals is None:
            return np.zeros(self.beamtransfer.ntel*self.telescope.nfreq, dtype=np.complex128)

        if vec.shape[0] != evecs.shape[0]:
            raise Exception("Vectors are incompatible.")

        # Construct the pseudo inverse
        invmodes = self.invmodes_m(mi, threshold)

        return np.dot(invmodes, vec)



    def project_vector_sky_to_kl(self, mi, vec, threshold=None):
        """Project an m-vector from the sky into the eigenbasis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [freq, pol, l]
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projvector : np.ndarray
            The vector projected into the eigenbasis.
        """
        tvec = self.beamtransfer.project_vector_sky_to_svd(mi, vec)

        return self.project_vector_svd_to_kl(mi, tvec, threshold)


    def project_matrix_svd_to_kl(self, mi, mat, threshold=None):
        """Project a matrix from the telescope basis into the eigenbasis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Telescope matrix to project.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projmatrix : np.ndarray
            The matrix projected into the eigenbasis.
        """
        evals, evecs = self.modes_m(mi, threshold)

        if (mat.shape[0] != evecs.shape[1]) or (mat.shape[0] != mat.shape[1]):
            raise Exception("Matrix size incompatible.")

        return np.dot(np.dot(evecs, mat), evecs.T.conj())


    def project_matrix_sky_to_kl(self, mi, mat, threshold=None):
        """Project a covariance matrix from the sky into the eigenbasis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mat : np.ndarray
            Sky matrix to project.
        threshold : real scalar, optional
            Returns only KL-modes with S/N greater than threshold. By default
            return all modes saved in the file (this maybe be a subset already,
            see `transform_save`).

        Returns
        -------
        projmatrix : np.ndarray
            The matrix projected into the eigenbasis.
        """


        mproj = self.beamtransfer.project_matrix_sky_to_svd(mi, mat)

        return self.project_matrix_svd_to_kl(mi, mproj, threshold)


    def project_sky_matrix_forward_old(self, mi, mat, threshold=None):

        npol = self.telescope.num_pol_sky
        lside = self.telescope.lmax + 1
        nfreq = self.telescope.nfreq

        st = time.time()

        evsky = self.skymodes_m(mi, threshold).reshape((-1, nfreq, npol, lside))
        et = time.time()

        #print "Evsky: %f" % (et-st)

        st = time.time()
        ev1n = np.transpose(evsky, (2, 3, 0, 1)).copy()
        ev1h = np.transpose(evsky, (2, 3, 1, 0)).conj()
        matf = np.zeros((evsky.shape[0], evsky.shape[0]), dtype=np.complex128)

        for pi in range(npol):
            for pj in range(npol):
                for li in range(lside):
                    matf += np.dot(np.dot(ev1n[pi, li], mat[pi, pj, li]), ev1h[pj, li])

        et = time.time()

        #print "Rest: %f" % (et-st)


        return matf



    def project_sky(self, sky, mlist = None, threshold=None, harmonic=False):

        # Set default list of m-modes (i.e. all of them), and partition
        if mlist is None:
            mlist = range(self.telescope.mmax + 1)
        mpart = mpiutil.partition_list_mpi(mlist)

        # Total number of sky modes.
        nmodes = self.beamtransfer.nfreq * self.beamtransfer.ntel

        # If sky is alm fine, if not perform spherical harmonic transform.
        alm = sky if harmonic else hputil.sphtrans_sky(sky, lmax=self.telescope.lmax)


        ## Routine to project sky onto eigenmodes
        def _proj(mi):
            p1 = self.project_sky_vector_forward(mi, alm[:, :, mi], threshold)
            p2 = np.zeros(nmodes, dtype=np.complex128)
            p2[-p1.size:] = p1
            return p2

        # Map over list of m's and project sky onto eigenbasis
        proj_sec = [(mi, _proj(mi)) for mi in mpart]

        # Gather projections onto the rank=0 node.
        proj_all = mpiutil.world.gather(proj_sec, root=0)

        proj_arr = None

        if mpiutil.rank0:
            # Create array to put projections into
            proj_arr = np.zeros((2*self.telescope.mmax + 1, nmodes), dtype=np.complex128)

            # Iterate over all gathered projections and insert into the array
            for proc_rank in proj_all:
                for pm in proc_rank:
                    proj_arr[pm[0]] = pm[1]

        # Return the projections (rank=0) or None elsewhere.
        return proj_arr
