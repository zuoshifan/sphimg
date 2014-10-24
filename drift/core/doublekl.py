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

        # Fetch the covariance matrices to diagonalise
        nside = self.beamtransfer.ndof(mi)
        if rank0:
            print 'nside = ', nside

        # Ensure that number of SVD degrees of freedom is non-zero before proceeding
        if nside == 0:
            return np.array([]), np.array([[]]), np.array([[]]), { 'ac' : 0.0, 'f_evals' : np.array([]) }

        # Construct S and F matrices and regularise foregrounds
        self.use_thermal = False
        # cs, cn = [ cv.reshape(nside, nside) for cv in self.sn_covariance(mi) ]
        cs, cn, dist = self.sn_covariance(mi, comm)

        # Find joint eigenbasis and transformation matrix
        if rank0:
            print 'Start first KL transfom for m = %d...' % mi
        if dist:
            evals, evecs = su.eigh_gen(cs, cn)
            evecs = evecs.to_global_array() # no need Hermitian transpose
            # evals, evecs = rt.eigh(cs, cn)
            # evecs = evecs.to_global_array()
            # evecs = evecs.T.conj()
            ac = 0.0
        else:
            evals, evecs, ac = kltransform.eigh_gen(cs, cn)
            evecs = evecs.T.conj() # need Hermitian transpose
        if rank0:
            print 'First KL transfom for m = %d done.' % mi

        # Get the indices that extract the high S/F ratio modes
        ind = np.where(evals > self.foreground_threshold)

        # Construct evextra dictionary (holding foreground ratio)
        evextra = { 'ac' : ac, 'f_evals' : evals.copy() }

        # Construct inverse transformation if required
        inv = None
        if self.inverse:
            inv = kltransform.inv_gen(evecs).T

        # Construct the foreground removed subset of the space
        evals = evals[ind]
        evecs = evecs[ind]
        inv = inv[ind] if self.inverse else None
        if rank0:
            print "Modes with S/F > %f: %i of %i" % (self.foreground_threshold, evals.size, nside)

        if evals.size > 0:
            # Generate the full S and N covariances in the truncated basis
            self.use_thermal = True
            # cs, cn = [ cv.reshape(nside, nside) for cv in self.sn_covariance(mi) ]
            cs, cn, dist = self.sn_covariance(mi, comm)
            if rank0:
                print 'Start second KL transfom for m = %d...' % mi
            if dist:
                # ensure no precess will have empty section of evecs
                if evecs.shape[0] >= cs.block_shape[0] * comm.Get_size():
                    # distribute calculation
                    evecs = np.asfortranarray(evecs)
                    evecs = core.DistributedMatrix.from_global_array(evecs, rank=0, block_shape=cs.block_shape, context=cs.context)
                    cs = rt.dot(evecs, rt.dot(cs, evecs, transA='N', transB='C'), transA='N', transB='N')
                    cn = rt.dot(evecs, rt.dot(cn, evecs, transA='N', transB='C'), transA='N', transB='N')

                    # Find the eigenbasis and the transformation into it.
                    evals, evecs2 = su.eigh_gen(cs, cn)
                    evecs = rt.dot(evecs2, evecs, transA='N', transB='N') # NOTE: no Hermitian transpose to evecs2
                    evecs = evecs.to_global_array() # no need Hermitian transpose
                    evecs2 = evecs2.to_global_array().T.conj()
                    ac = 0.0
                else:
                    cs = cs.to_global_array()
                    cn = cn.to_global_array()
                    cs = np.dot(evecs, np.dot(cs, evecs.T.conj()))
                    cn = np.dot(evecs, np.dot(cn, evecs.T.conj()))

                    # Find the eigenbasis and the transformation into it.
                    evals, evecs2, ac = kltransform.eigh_gen(cs, cn)
                    evecs = np.dot(evecs2.T.conj(), evecs)
            else:
                cs = np.dot(evecs, np.dot(cs, evecs.T.conj()))
                cn = np.dot(evecs, np.dot(cn, evecs.T.conj()))

                # Find the eigenbasis and the transformation into it.
                evals, evecs2, ac = kltransform.eigh_gen(cs, cn)
                evecs = np.dot(evecs2.T.conj(), evecs)
            if rank0:
                print 'Second KL transfom for m = %d done.' % mi

            # Construct the inverse if required.
            if self.inverse:
                inv2 = kltransform.inv_gen(evecs2)
                inv = np.dot(inv2, inv)

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
            ta = np.zeros(shape, dtype=np.float64)

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

        mlist = range(self.telescope.mmax+1)
        shape = (2, self.beamtransfer.ndofmax)
        evarray = kltransform.collect_m_array(mlist, evfunc, shape, np.float64)

        if mpiutil.rank0:
            with h5py.File(self._all_evfile, 'w') as f:
                f.create_dataset('evals', data=evarray[:, 0], compression='lzf')
                f.create_dataset('f_evals', data=evarray[:, 1], compression='lzf')
