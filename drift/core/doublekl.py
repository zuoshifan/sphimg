import sys
import os

import numpy as np
import h5py

from drift.core import kltransform
from drift.util import mpiutil, config
from drift.util import typeutil

from scalapy import core
import scalapy.routines as rt
from drift.util import scalapyutil as su


class DoubleKL(kltransform.KLTransform):
    """Modified KL technique that performs a first transformation to remove
    foreground modes, and a subsequent transformation to diagonalise the full
    noise (remaining foregrounds+instrumental space).

    Attributes
    ----------
    foreground_threshold : scalar
        Ratio of S/F power below which we throw away modes as being foreground
        contaminated.
    """

    foreground_threshold = config.Property(proptype=typeutil.nonnegative_float, default=100.0)

    def _transform_m(self, mi, comm=None):

        rank0 = True if comm is None or comm.Get_rank() == 0 else False
        if rank0:
            print "Solving for Eigenvalues...."

        nside = self.beamtransfer.ndof(mi)
        if rank0:
            print 'nside = ', nside

        # Ensure that number of SVD degrees of freedom is non-zero before proceeding
        if nside == 0:
            return np.array([]), np.array([[]]), np.array([[]]), { 'ac' : 0.0, 'f_evals' : np.array([]) }

        # Construct S and F matrices and regularise foregrounds
        self.use_thermal = False
        # Fetch the covariance matrices to diagonalise
        cs, cn, dist = self.sn_covariance(mi, comm)

        # Find joint eigenbasis and transformation matrix
        if rank0:
            print 'Start first KL transfom for m = %d...' % mi
        if dist:
            # evals, evecs = su.eigh_gen(cs, cn)
            # evecs = evecs.to_global_array() # no need Hermitian transpose
            # evals, evecs = rt.eigh(cs, cn)
            evals, evecs, ac = kltransform.dist_eigh_gen(cs, cn)
            evecs = evecs.H # Hermitian conjugate of the distributed matrix
            # ac = 0.0
        else:
            if rank0:
                evals, evecs, ac = kltransform.eigh_gen(cs, cn)
                evecs = evecs.T.conj() # need Hermitian transpose
            else:
                evals, evecs, ac = None, None, 0.0
        if rank0:
            print 'First KL transfom for m = %d done.' % mi
            sys.stdout.flush()

        # Construct evextra dictionary (holding foreground ratio)
        if rank0:
            evextra = { 'ac' : ac, 'f_evals' : evals.copy() }
        else:
            evextra = None

        # Get the indices that extract the high S/F ratio modes
        if rank0:
            ind = np.searchsorted(evals, self.foreground_threshold)
        else:
            ind = 0
        if comm is not None:
            ind = comm.bcast(ind, root=0)
        if rank0:
            print "Modes with S/F > %f: %i of %i" % (self.foreground_threshold, nside-ind, nside)

        # if no evals greater than foreground_threshold, directly return, no need to do inverse and other computations
        if ind == nside:
            if comm is not None:
                comm.Barrier()
            return np.array([]), np.array([]).reshape(0, nside), np.array([]).reshape(0, nside), evextra


        # else one or more evals greater than foreground_threshold
        if rank0:
            print 'Start inverse calculation for m = %d...' % mi
        # Construct inverse transformation if required
        inv = None
        if self.inverse:
            if dist:
                inv = rt.pinv(evecs, overwrite_a=False).T # NOTE: must overwrite_a = False
            else:
                inv = kltransform.inv_gen(evecs).T if evecs is not None else None
        if rank0:
            print 'Inverse calculation for m = %d done.' % mi

        if dist:
            # copy to numpy array
            evals = evals[ind:]
            dtype = evecs.dtype
            if ind < nside:
                evecs = evecs.self2np(srow=ind, scol=0, rank=0)
            else:
                evecs = np.array([], dtype=dtype).reshape(0, nside)
            if self.inverse:
                if ind < nside:
                    inv = inv.self2np(srow=ind, scol=0, rank=0)
                else:
                    inv = np.array([], dtype=dtype).reshape(0, nside)
        else:
            if rank0:
                # Construct the foreground removed subset of the space
                evals = evals[ind:]
                evecs = evecs[ind:]
                inv = inv[ind:] if self.inverse else None

        if rank0 and evals.size > 0:
            # Generate the full S and N covariances in the truncated basis
            cs = np.diag(evals) # Lambda_s
            cn = np.dot(evecs, evecs.T.conj()) # P_s Nbar P_s^\dagger, where Nbar = I
            cn[np.diag_indices_from(cn)] += 1.0 # I + P_s P_s^\dagger
            print 'Start second KL transfom for m = %d...' % mi
            # Find the eigenbasis and the transformation into it.
            evals, evecs2, ac = kltransform.eigh_gen(cs, cn)
            evecs = np.dot(evecs2.T.conj(), evecs)
            print 'Second KL transfom for m = %d done.' % mi
            sys.stdout.flush()

            # Construct the inverse if required.
            if self.inverse:
                inv2 = kltransform.inv_gen(evecs2)
                inv = np.dot(inv2, inv)

        if comm is not None:
            comm.Barrier()

        return evals, evecs, inv, evextra


    def _ev_save_hook(self, f, evextra):

        kltransform.KLTransform._ev_save_hook(self, f, evextra)

        # Save out S/F ratios
        f.create_dataset('f_evals', data=evextra['f_evals'], compression='lzf')


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
            ta = np.zeros((2, self.beamtransfer.ndofmax), dtype=np.float64)

            # ensure that data files has already been saved to disk at the time of reading (file I/O is much slower than CPU)
            while True:
                try:
                    with h5py.File(self._evfile(mi), 'r') as f:
                        if f['evals_full'].shape[0] > 0:
                            ev = f['evals_full'][:]
                            fev = f['f_evals'][:]
                            ta[0, -ev.size:] = ev
                            ta[1, -fev.size:] = fev

                    break
                except IOError:
                    pass

            return ta

        mis = self.telescope.mmax + 1
        shape = (2, self.beamtransfer.ndofmax)
        evarray = kltransform.collect_m_array(mis, evfunc, shape, np.float64)

        if mpiutil.rank0:
            with h5py.File(self._all_evfile, 'w') as f:
                f.create_dataset('evals', data=evarray[:, 0], compression='lzf')
                f.create_dataset('f_evals', data=evarray[:, 1], compression='lzf')

        mpiutil.barrier()