try:
   import cPickle as pickle
except ImportError:
   import pickle

import os
import shutil

import h5py
import healpy
import numpy as np

from cora.util import hputil

from drift.core import manager, kltransform
from drift.util import util, mpiutil


def smoothalm(alms, fwhm=0.0, sigma=None, invert=False, mmax=None, verbose=True, inplace=True):
    """Smooth alm with a Gaussian symmetric beam function.

    Parameters
    ----------
    alms : np.ndarray[freq, pol, l, m]
        The array of alms.
    fwhm, sigma, invert, mmax, verbose, inplace :
        See the corresponding paramters of `healpy.sphtfunc.smoothalm`.

    Returns
    -------
    alms: np.ndarray[freq, pol, l, m]
        The array of alms after smoothing with a Gaussian symmetric beam function.

    Notes
    -----
    See _[1].

    .. [1] Seon, 2006. Smoothing of an All-Sky Survey Map with a Fisher-Von Mises Function.
    """
    nfreq = alms.shape[0]
    npol = alms.shape[1]
    lmax = alms.shape[2] -1
    for fi in range(nfreq):
        almp = [hputil.pack_alm(alms[fi, ipol]) for ipol in range(npol)]
        if npol == 1:
            alms[fi, 0, :, :] = hputil.unpack_alm(healpy.sphtfunc.smoothalm(almp, fwhm=fwhm, sigma=sigma, invert=invert, pol=False, mmax=mmax, verbose=verbose, inplace=inplace), lmax=lmax)
        elif npol == 3 or npol == 4:
            alms[fi, :3, :, :] = [hputil.unpack_alm(alm, lmax=lmax) for alm in healpy.sphtfunc.smoothalm(almp[:3], fwhm=fwhm, sigma=sigma, invert=invert, pol=True, mmax=mmax, verbose=verbose, inplace=inplace)]
            if npol == 4:
                alms[fi, 3, :, :] = hputil.unpack_alm(healpy.sphtfunc.smoothalm(almp[3], fwhm=fwhm, sigma=sigma, invert=invert, pol=False, mmax=mmax, verbose=verbose, inplace=inplace), lmax=lmax)
        else:
            raise Exception('Unexpected npol = %d.' % npol)

    return alms


class Timestream(object):

    directory = None
    output_directory = None
    beamtransfer_dir = None

    no_m_zero = True




    #============ Constructor etc. =====================

    def __init__(self, tsdir, tsname, prodmanager):
        """Create a new Timestream object.

        Parameters
        ----------
        tsdir : string
            Directory to create the Timestream in.
        prodmanager : drift.core.manager.ProductManager
            ProductManager object containing the analysis products.
        """
        self.directory = os.path.abspath(tsdir)
        self.output_directory = self.directory
        self.tsname = tsname
        self.manager = prodmanager

    #====================================================


    #===== Accessing the BeamTransfer and Telescope =====

    _beamtransfer = None
    @property
    def beamtransfer(self):
        """The BeamTransfer object corresponding to this timestream.
        """

        return self.manager.beamtransfer

    @property
    def telescope(self):
        """The telescope object corresponding to this timestream.
        """
        return self.beamtransfer.telescope

    #====================================================


    #======== Fetch and generate the f-stream ===========


    @property
    def _fdir(self):
        return self.directory + '/timestream_f/'


    def _ffile(self, fi):
        # Pattern to form the `freq` ordered file.
        pat = self._fdir + "timestream_%s.hdf5" % util.natpattern(self.telescope.nfreq)
        return pat % fi


    @property
    def _fcommondata_file(self):
        return self._fdir + 'ts_commondata.hdf5'


    @property
    def ntime(self):
        """Get the number of timesamples."""

        with h5py.File(self._fcommondata_file, 'r') as f:
            ntime = f.attrs['ntime']

        return ntime


    def timestream_f(self, fi):
        """Fetch the timestream for a given frequency.

        Parameters
        ----------
        fi : integer
            Frequency to load.

        Returns
        -------
        timestream : np.ndarray[npairs, ntime]
            The visibility timestream.
        """

        with h5py.File(self._ffile(fi), 'r') as f:
            ts = f['timestream'][:]
        return ts

    #====================================================


    #======== Fetch and generate the m-modes ============

    @property
    def _mdir(self):
        return self.output_directory + '/mmodes/'


    def _mfile(self, mi):
        # Pattern to form the `m` ordered file.
        pat = self._mdir + 'mode_%s.hdf5' % util.natpattern(self.telescope.mmax)
        return pat % abs(mi)


    def mmode(self, mi):
        """Fetch the timestream m-mode for a specified m.

        Parameters
        ----------
        mi : integer
            m-mode to load.

        Returns
        -------
        timestream : np.ndarray[nfreq, pm, npairs]
            The visibility m-modes.
        """

        with h5py.File(self._mfile(mi), 'r') as f:
            return f['mmode'][:]


    def generate_mmodes(self):
        """Calculate the m-modes corresponding to the Timestream.

        Perform an MPI transpose for efficiency.
        """

        completed_file = self._mdir + 'COMPLETED_M'
        if os.path.exists(completed_file):
            if mpiutil.rank0:
                print "******* m-files already generated ********"
            mpiutil.barrier()
            return

        tel = self.telescope
        mmax = tel.mmax
        nfreq = tel.nfreq

        lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
        lm, sm, em = mpiutil.split_local(mmax + 1)

        # Load in the local frequencies of the time stream
        tstream = np.zeros((lfreq, tel.npairs, self.ntime), dtype=np.complex128)
        for lfi, fi in enumerate(range(sfreq, efreq)):
            tstream[lfi] = self.timestream_f(fi)

        # FFT to calculate the m-modes for the timestream
        row_mmodes = np.fft.fft(tstream, axis=-1) / self.ntime

        ## Combine positive and negative m parts.
        row_mpairs = np.zeros((lfreq, 2, tel.npairs, mmax+1), dtype=np.complex128)

        row_mpairs[:, 0, ..., 0] = row_mmodes[..., 0]
        for mi in range(1, mmax+1):
            row_mpairs[:, 0, ..., mi] = row_mmodes[...,  mi]
            row_mpairs[:, 1, ..., mi] = row_mmodes[..., -mi].conj()

        # Transpose to get the entirety of an m-mode on each process (i.e. all frequencies)
        col_mmodes = mpiutil.transpose_blocks(row_mpairs, (nfreq, 2, tel.npairs, mmax + 1))

        # Transpose the local section to make the m's first
        col_mmodes = np.transpose(col_mmodes, (3, 0, 1, 2))

        for lmi, mi in enumerate(range(sm, em)):

            # Make directory for m-mode
            try:
                if not os.path.exists(self._mdir):
                    os.makedirs(self._mdir)
            except OSError:
                pass

            with h5py.File(self._mfile(mi), 'w') as f:
                f.create_dataset('/mmode', data=col_mmodes[lmi], compression='lzf')
                f.attrs['m'] = mi

        if mpiutil.rank0:

            # Make file marker that the m's have been correctly generated:
            open(completed_file, 'a').close()

        mpiutil.barrier()

    #====================================================


    #======== Make and fetch SVD m-modes ================

    @property
    def _svddir(self):
        return self.output_directory + '/svdmodes/'

    def _svdfile(self, mi):
        # Pattern to form the `m` ordered file.
        pat = self._svddir + 'svd_%s.hdf5' % util.natpattern(self.telescope.mmax)
        return pat % abs(mi)


    def mmode_svd(self, mi):
        """Fetch the SVD m-mode for a specified m.

        Parameters
        ----------
        mi : integer
            m-mode to load.

        Returns
        -------
        svd_mode : np.ndarray[nfreq, pm, npairs]
            The visibility m-modes.
        """

        with h5py.File(self._svdfile(mi), 'r') as f:
            if f['mmode_svd'].shape[0] == 0:
                return np.zeros((0,), dtype=np.complex128)
            else:
                return f['mmode_svd'][:]


    def generate_mmodes_svd(self):
        """Generate the SVD modes for the Timestream.
        """

        completed_file = self._svddir + 'COMPLETED_SVD'
        if os.path.exists(completed_file):
            if mpiutil.rank0:
                print "******* svd-files already generated ********"
            mpiutil.barrier()
            return

        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            # Make directory for svd-mode
            try:
                if not os.path.exists(self._svddir):
                    os.makedirs(self._svddir)
            except OSError:
                pass

            if os.path.exists(self._svdfile(mi)):
            # if mi in completed_mlist:
                print "File %s exists. Skipping..." % self._svdfile(mi)
                continue

            tm = self.mmode(mi).reshape(self.telescope.nfreq, 2*self.telescope.npairs)
            svdm = self.beamtransfer.project_vector_telescope_to_svd(mi, tm)

            with h5py.File(self._svdfile(mi), 'w') as f:
                f.create_dataset('mmode_svd', data=svdm, compression='lzf')
                f.attrs['m'] = mi

        if mpiutil.rank0:

            # Make file marker that the m's have been correctly generated:
            open(completed_file, 'a').close()

        mpiutil.barrier()


    #====================================================


    @property
    def _mapsdir(self):
        return self.output_directory + '/maps/'

    #======== Make map from uncleaned stream ============

    def mapmake_full(self, nside, mapname, fwhm=0.0, rank_ratio=0.0, lcut=None):

        mapfile = self._mapsdir + mapname

        if os.path.exists(mapfile):
            if mpiutil.rank0:
                print "File %s exists. Skipping..." % mapfile
            mpiutil.barrier()
            return

        def _make_alm(mi):

            print "Making %i" % mi

            mmode = self.mmode(mi)
            sphmode = self.beamtransfer.project_vector_telescope_to_sky(mi, mmode, rank_ratio, lcut)

            return sphmode

        alm_list = mpiutil.parallel_map(_make_alm, range(self.telescope.mmax + 1))

        if mpiutil.rank0:

            alm = np.zeros((self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1,
                            self.telescope.lmax + 1), dtype=np.complex128)

            for mi in range(self.telescope.mmax + 1):

                alm[..., mi] = alm_list[mi]

            # Smooth alm with a Gaussian symmetric beam function.
            if fwhm > 0.0:
                alm = smoothalm(alm, fwhm=fwhm)

            skymap = hputil.sphtrans_inv_sky(alm, nside)

            # Make directory for maps file
            if not os.path.exists(self._mapsdir):
                os.makedirs(self._mapsdir)

            with h5py.File(mapfile, 'w') as f:
                f.create_dataset('/map', data=skymap)

        mpiutil.barrier()


    def mapmake_svd(self, nside, mapname, fwhm=0.0, rank_ratio=0.0, lcut=None):

        mapfile = self._mapsdir + mapname

        if os.path.exists(mapfile):
            if mpiutil.rank0:
                print "File %s exists. Skipping..." % mapfile
            mpiutil.barrier()
            return

        self.generate_mmodes_svd()

        def _make_alm(mi):

            print "Making %i" % mi

            svdmode = self.mmode_svd(mi)
            sphmode = self.beamtransfer.project_vector_svd_to_sky(mi, svdmode, rank_ratio, lcut)

            return sphmode

        alm_list = mpiutil.parallel_map(_make_alm, range(self.telescope.mmax + 1))

        if mpiutil.rank0:

            alm = np.zeros((self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1,
                            self.telescope.lmax + 1), dtype=np.complex128)

            for mi in range(self.telescope.mmax + 1):

                alm[..., mi] = alm_list[mi]

            # Smooth alm with a Gaussian symmetric beam function.
            if fwhm > 0.0:
                alm = smoothalm(alm, fwhm=fwhm)

            skymap = hputil.sphtrans_inv_sky(alm, nside)

            # Make directory for maps file
            if not os.path.exists(self._mapsdir):
                os.makedirs(self._mapsdir)

            with h5py.File(mapfile, 'w') as f:
                f.create_dataset('/map', data=skymap)

        mpiutil.barrier()

    #====================================================


    #========== Project into KL-mode basis ==============

    def set_kltransform(self, klname, threshold=None):

        self.klname = klname
        kl = self.manager.kltransforms[self.klname]

        if threshold is None:
            threshold = kl.threshold
        self.klthreshold = threshold

        try:
            self.foreground_threshold = kl.foreground_threshold
        except AttributeError:
            self.foreground_threshold = None


    @property
    def _kldir(self):
        if self.foreground_threshold is None:
            return self.output_directory + '/klmodes/' + self.klname + '_%f/' % self.klthreshold
        else:
            return self.output_directory + '/klmodes/' + self.klname + '_%f_%f/' % (self.klthreshold, self.foreground_threshold)

    def _klfile(self, mi):
        # Pattern to form the `m` ordered file.
        pat = self._kldir + '%s_mode_%s.hdf5' % (self.klname, util.natpattern(self.telescope.mmax))
        return pat % (mi)

    @property
    def _klmodes_file(self):
        if self.foreground_threshold is None:
            return os.path.abspath(os.path.join(self._kldir, os.pardir)) + '/%s_modes_%f.hdf5' % (self.klname, self.klthreshold)
        else:
            return os.path.abspath(os.path.join(self._kldir, os.pardir)) + '/%s_modes_%f_%f.hdf5' % (self.klname, self.klthreshold, self.foreground_threshold)


    def mmode_kl(self, mi):
        with h5py.File(self._klfile(mi), 'r') as f:
            if f['mmode_kl'].shape[0] == 0:
                return np.zeros((0,), dtype=np.complex128)
            else:
                return f['mmode_kl'][:]


    def generate_mmodes_kl(self):
        """Generate the KL modes for the Timestream.
        """

        kl = self.manager.kltransforms[self.klname]

        completed_file = self._kldir + 'COMPLETED_%s' % self.klname.upper()
        if os.path.exists(completed_file):
            if mpiutil.rank0:
                print "******* %s-files already generated ********" % self.klname
            mpiutil.barrier()
            return

        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            # Make directory for kl-mode
            try:
                if not os.path.exists(self._kldir):
                    os.makedirs(self._kldir)
            except OSError:
                pass

            if os.path.exists(self._klfile(mi)):
            # if mi in completed_mlist:
                print "File %s exists. Skipping..." % self._klfile(mi)
                continue

            svdm = self.mmode_svd(mi) #.reshape(self.telescope.nfreq, 2*self.telescope.npairs)
            klm = kl.project_vector_svd_to_kl(mi, svdm, threshold=self.klthreshold)

            with h5py.File(self._klfile(mi), 'w') as f:
                f.create_dataset('mmode_kl', data=klm)
                f.attrs['m'] = mi

        if mpiutil.rank0:

            # Make file marker that the m's have been correctly generated:
            open(completed_file, 'a').close()

        mpiutil.barrier()


    def collect_mmodes_kl(self):

        if os.path.exists(self._klmodes_file):
            if mpiutil.rank0:
                print
                print '=' * 80
                print "File: %s exists. Skipping..." % self._klmodes_file
            mpiutil.barrier()
            return

        def evfunc(mi):
            evf = np.zeros(self.beamtransfer.ndofmax, dtype=np.complex128)

            ev = self.mmode_kl(mi)
            if ev.size > 0:
                evf[-ev.size:] = ev

            return evf


        mlist = range(self.telescope.mmax+1)
        shape = (self.beamtransfer.ndofmax, )
        evarray = kltransform.collect_m_array(mlist, evfunc, shape, np.complex128)

        if mpiutil.rank0:
            print
            print '=' * 80
            print "Creating all kl-modes file %s..." % self._klmodes_file
            with h5py.File(self._klmodes_file, 'w') as f:
                f.create_dataset('evals', data=evarray, compression='lzf')

        mpiutil.barrier()


    def fake_kl_data(self):

        kl = self.manager.kltransforms[self.klname]

        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1):

            evals = kl.evals_m(mi)

            if evals is None:
                klmode = np.array([], dtype=np.complex128)
            else:
                modeamp = ((evals + 1.0) / 2.0)**0.5
                klmode = modeamp * (np.array([1.0, 1.0J]) * np.random.standard_normal((modeamp.shape[0], 2))).sum(axis=1)


            with h5py.File(self._klfile(mi), 'w') as f:
                f.create_dataset('mmode_kl', data=klmode)
                f.attrs['m'] = mi

        mpiutil.barrier()


    def mapmake_kl(self, nside, mapname, wiener=False,  rank_ratio=0.0, lcut=None):

        mapfile = self._mapsdir + mapname

        if os.path.exists(mapfile):
            if mpiutil.rank0:
                print "File %s exists. Skipping..." % mapfile
            mpiutil.barrier()
            return

        kl = self.manager.kltransforms[self.klname]

        if not kl.inverse:
            raise Exception("Need the inverse to make a meaningful map.")

        def _make_alm(mi):
            print "Making %i" % mi

            klmode = self.mmode_kl(mi)

            if wiener:
                evals = kl.evals_m(mi, self.klthreshold)

                if evals is not None:
                    klmode *= (evals / (1.0 + evals))

            isvdmode = kl.project_vector_kl_to_svd(mi, klmode, threshold=self.klthreshold)

            sphmode = self.beamtransfer.project_vector_svd_to_sky(mi, isvdmode, rank_ratio, lcut)

            return sphmode

        alm_list = mpiutil.parallel_map(_make_alm, range(self.telescope.mmax + 1))

        if mpiutil.rank0:

            alm = np.zeros((self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1,
                            self.telescope.lmax + 1), dtype=np.complex128)

            # Determine whether to use m=0 or not
            mlist = range(1 if self.no_m_zero else 0, self.telescope.mmax + 1)

            for mi in mlist:

                alm[..., mi] = alm_list[mi]

            skymap = hputil.sphtrans_inv_sky(alm, nside)

            # Make directory for maps file
            if not os.path.exists(self._mapsdir):
                os.makedirs(self._mapsdir)

            with h5py.File(mapfile, 'w') as f:
                f.create_dataset('/map', data=skymap)

        mpiutil.barrier()

    #====================================================


    #======= Estimate powerspectrum from data ===========

    @property
    def _psdir(self):
        return self.output_directory + '/ps/'

    @property
    def _psfile(self):
        # Pattern to form the `m` ordered file.
        return self._psdir + 'ps_%s.hdf5' % self.psname



    def set_psestimator(self, psname):
        self.psname = psname


    def powerspectrum(self):

        if os.path.exists(self._psfile):
            if mpiutil.rank0:
                print "File %s exists. Skipping..." % self._psfile
            mpiutil.barrier()
            return

        import scipy.linalg as la

        ps = self.manager.psestimators[self.psname]
        ps.load_clarray()

        def _q_estimate(mi):

            return ps.q_estimator(mi, self.mmode_kl(mi))

        # Determine whether to use m=0 or not
        mlist = range(1 if self.no_m_zero else 0, self.telescope.mmax + 1)
        qvals = mpiutil.parallel_map(_q_estimate, mlist)

        # Delete cache of bands for memory reasons
        ps.delbands()

        qtotal = np.array(qvals).sum(axis=0)

        fisher, bias = ps.fisher_bias()

        # Get mixing matrix
        with h5py.File(ps._psfile, 'r') as f:
           uw_mm = f['uw_mmatrix'][...] # Unwindowed mixing matrix
           uc_mm = f['uc_mmatrix'][...] # Uncorrelated mixing matrix
           mv_mm = f['mv_mmatrix'][...] # Minimum variance mixing matrix
           iv_mm = f['iv_mmatrix'][...] # Inverse variance mixing matrix

        # Powerspectrum estimator
        uw_powerspectrum = np.dot(uw_mm, qtotal - bias) # Unwindowed power spectrum
        uc_powerspectrum = np.dot(uc_mm, qtotal - bias) # Uncorrelated power spectrum
        mv_powerspectrum = np.dot(mv_mm, qtotal - bias) # Minimum variance power spectrum
        iv_powerspectrum = np.dot(iv_mm, qtotal - bias) # Inverse variance power spectrum


        if mpiutil.rank0:
            # make directory for power spectrum files
            if not os.path.exists(self._psdir):
                os.makedirs(self._psdir)

            # copy ps file from product directory
            shutil.copyfile(ps._psfile, self._psfile)

            with h5py.File(self._psfile, 'a') as f:

                f.create_dataset('uw_powerspectrum', data=uw_powerspectrum.real)
                f.create_dataset('uc_powerspectrum', data=uc_powerspectrum.real)
                f.create_dataset('mv_powerspectrum', data=mv_powerspectrum.real)
                f.create_dataset('iv_powerspectrum', data=iv_powerspectrum.real)

        mpiutil.barrier()



    #====================================================


    #======== Load and save the Pickle files ============

    def __getstate__(self):
        ## Remove the attributes we don't want pickled.
        state = self.__dict__.copy()

        for key in self.__dict__:
            #if (key in delkeys) or (key[0] == "_"):
            if (key[0] == "_"):
                del state[key]

        return state


    @property
    def _picklefile(self):
        # The filename for the pickled telescope
        return self.output_directory + "/timestreamobject.pickle"


    def save(self):
        """Save out the Timestream object information."""

        # Save pickled telescope object
        print
        print '=' * 80
        print "Saving Timestream object %s..." % self.tsname
        with open(self._picklefile, 'w') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, tsdir):
        """Load the Timestream object from disk.

        Parameters
        ----------
        tsdir : string
            Name of the directory containing the Timestream object.
        """

        # Create temporary object to extract picklefile property
        tmp_obj = cls(tsdir, tsdir)

        with open(tmp_obj._picklefile, 'r') as f:
            print "=== Loading Timestream object. ==="
            return pickle.load(f)

    #====================================================



def cross_powerspectrum(timestreams, psname, psfile):

    if os.path.exists(psfile):
        if mpiutil.rank0:
            print "File %s exists. Skipping..." % psfile
        mpiutil.barrier()
        return

    import scipy.linalg as la

    products = timestreams[0].manager

    ps = products.psestimators[psname]
    ps.load_clarray()

    nstream = len(timestreams)

    def _q_estimate(mi):

        # qp = np.zeros((nstream, nstream, ps.nbands), dtype=np.float64)
        qp = np.zeros((nstream, nstream, ps.nbands), dtype=np.complex128)

        for ti in range(nstream):
            for tj in range(ti+1, nstream):

                print "Making m=%i (%i, %i)" % (mi, ti, tj)

                si = timestreams[ti]
                sj = timestreams[tj]

                qp[ti, tj] = ps.q_estimator(mi, si.mmode_kl(mi), sj.mmode_kl(mi))
                # qp[tj, ti] = qp[ti, tj]
                qp[tj, ti] = np.conj(qp[ti, tj])

        return qp

    # Determine whether to use m=0 or not
    mlist = range(1 if timestreams[0].no_m_zero else 0, products.telescope.mmax + 1)
    qvals = mpiutil.parallel_map(_q_estimate, mlist)

    # Delete cache of bands for memory reasons
    ps.delbands()

    qtotal = np.array(qvals).sum(axis=0)

    fisher, bias = ps.fisher_bias()

    # Subtract bias and reshape into new array
    qtotal = (qtotal - bias).reshape(nstream**2, ps.nbands).T

    powerspectrum =  np.dot(la.inv(fisher), qtotal)
    powerspectrum = powerspectrum.T.reshape(nstream, nstream, ps.nbands)


    if mpiutil.rank0:
        # make directory for cross power spectrum files
        if not os.path.exists(os.path.dirname(psfile)):
            os.makedirs(os.path.dirname(psfile))

        # copy ps file from product directory
        shutil.copyfile(ps._psfile, psfile)

        with h5py.File(psfile, 'a') as f:
            f.create_dataset('powerspectrum', data=powerspectrum.real)

    mpiutil.barrier()



# kwargs is to absorb any extra params
def simulate(m, outdir, tsname, maps=[], ndays=None, resolution=0, seed=None, **kwargs):
    """Create a simulated timestream and save it to disk.

    Parameters
    ----------
    m : ProductManager object
        Products of telescope to simulate.
    outdir : directoryname
        Directory that we will save the timestream into.
    maps : list
        List of map filenames. The sum of these form the simulated sky.
    ndays : int, optional
        Number of days of observation. Setting `ndays = None` (default) uses
        the default stored in the telescope object; `ndays = 0`, assumes the
        observation time is infinite so that the noise is zero.
    resolution : scalar, optional
        Approximate time resolution in seconds. Setting `resolution = 0`
        (default) calculates the value from the mmax.

    Returns
    -------
    timestream : Timestream
    """

    # Create timestream object
    tstream = Timestream(outdir, tsname, m)

    completed_file = tstream._fdir + 'COMPLETED_TIMESTREAM'
    if os.path.exists(completed_file):
        if mpiutil.rank0:
            print "******* timestream-files already generated ********"
        mpiutil.barrier()
        return

    ## Read in telescope system
    bt = m.beamtransfer
    tel = bt.telescope

    lmax = tel.lmax
    mmax = tel.mmax
    nfreq = tel.nfreq
    npol = tel.num_pol_sky

    projmaps = (len(maps) > 0)

    lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
    local_freq = range(sfreq, efreq)

    lm, sm, em = mpiutil.split_local(mmax + 1)

    # If ndays is not set use the default value.
    if ndays is None:
        ndays = tel.ndays

    # Calculate the number of timesamples from the resolution
    if resolution == 0:
        # Set the minimum resolution required for the sky.
        ntime = 2*mmax+1
    else:
        # Set the cl
        ntime = int(np.round(24 * 3600.0 / resolution))


    col_vis = np.zeros((tel.npairs, lfreq, ntime), dtype=np.complex128)

    ## If we want to add maps use the m-mode formalism to project a skymap
    ## into visibility space.

    if projmaps:

        # Load file to find out the map shapes.
        with h5py.File(maps[0], 'r') as f:
            mapshape = f['map'].shape

        if lfreq > 0:

            # Allocate array to store the local frequencies
            row_map = np.zeros((lfreq,) + mapshape[1:], dtype=np.float64)

            # Read in and sum up the local frequencies of the supplied maps.
            for mapfile in maps:
                with h5py.File(mapfile, 'r') as f:
                    row_map += f['map'][sfreq:efreq]

            # Calculate the alm's for the local sections
            row_alm = hputil.sphtrans_sky(row_map, lmax=lmax).reshape((lfreq, npol * (lmax+1), lmax+1))

        else:
            row_alm = np.zeros((lfreq, npol * (lmax+1), lmax+1), dtype=np.complex128)

        # mpiutil.barrier()

        # Perform the transposition to distribute different m's across processes. Neat
        # tip, putting a shorter value for the number of columns, trims the array at
        # the same time
        col_alm = mpiutil.transpose_blocks(row_alm, (nfreq, npol * (lmax+1), mmax+1))

        # Transpose and reshape to shift m index first.
        col_alm = np.transpose(col_alm, (2, 0, 1)).reshape(lm, nfreq, npol, lmax+1)

        # Create storage for visibility data
        vis_data = np.zeros((lm, nfreq, bt.ntel), dtype=np.complex128)

        # Iterate over m's local to this process and generate the corresponding
        # visibilities
        for mp, mi in enumerate(range(sm, em)):
            vis_data[mp] = bt.project_vector_sky_to_telescope(mi, col_alm[mp])

        # Rearrange axes such that frequency is last (as we want to divide
        # frequencies across processors)
        row_vis = vis_data.transpose((0, 2, 1))#.reshape((lm * bt.ntel, nfreq))

        # Parallel transpose to get all m's back onto the same processor
        col_vis_tmp = mpiutil.transpose_blocks(row_vis, ((mmax+1), bt.ntel, nfreq))
        col_vis_tmp = col_vis_tmp.reshape(mmax + 1, 2, tel.npairs, lfreq)

        # Transpose the local section to make the m's the last axis and unwrap the
        # positive and negative m at the same time.
        col_vis[..., 0] = col_vis_tmp[0, 0]
        for mi in range(1, mmax+1):
            col_vis[...,  mi] = col_vis_tmp[mi, 0]
            col_vis[..., -mi] = col_vis_tmp[mi, 1].conj()  # Conjugate only (not (-1)**m - see paper)


        del col_vis_tmp

    ## If we're simulating noise, create a realisation and add it to col_vis
    if ndays > 0 and lfreq > 0:

        # Fetch the noise powerspectrum
        noise_ps = tel.noisepower(np.arange(tel.npairs)[:, np.newaxis], np.array(local_freq)[np.newaxis, :], ndays=ndays).reshape(tel.npairs, lfreq)[:, :, np.newaxis]


        # Seed random number generator to give consistent noise
        if seed is not None:
            # Must include rank such that we don't have massive power deficit from correlated noise
            np.random.seed(seed + mpiutil.rank)

        # Create and weight complex noise coefficients
        noise_vis = (np.array([1.0, 1.0J]) * np.random.standard_normal(col_vis.shape + (2,))).sum(axis=-1)
        noise_vis *= (noise_ps / 2.0)**0.5

        # Reset RNG
        if seed is not None:
            np.random.seed()

        # Add into main noise sims
        col_vis += noise_vis

        del noise_vis


    # Fourier transform m-modes back to get timestream.
    vis_stream = np.fft.ifft(col_vis, axis=-1) * ntime
    vis_stream = vis_stream.reshape(tel.npairs, lfreq, ntime)

    # The time samples the visibility is calculated at
    tphi = np.linspace(0, 2*np.pi, ntime, endpoint=False)

    ## Iterate over the local frequencies and write them to disk.
    for lfi, fi in enumerate(local_freq):

        # Make directory if required
        try:
            if not os.path.exists(tstream._fdir):
                os.makedirs(tstream._fdir)
        except OSError:
            pass

        # Write file contents
        with h5py.File(tstream._ffile(fi), 'w') as f:

            # Timestream data
            f.create_dataset('/timestream', data=vis_stream[:, lfi], compression='lzf')

    # mpiutil.barrier()

    if mpiutil.rank0:
        print
        print '=' * 80
        print 'Creating time stream %s common data file %s...' % (tstream.tsname, tstream._fcommondata_file)
        # Write common data
        with h5py.File(tstream._fcommondata_file, 'w') as f:
            f.create_dataset('/phi', data=tphi)

            # Telescope layout data
            f.create_dataset('/feedmap', data=tel.feedmap)
            f.create_dataset('/feedconj', data=tel.feedconj)
            f.create_dataset('/feedmask', data=tel.feedmask)
            f.create_dataset('/uniquepairs', data=tel.uniquepairs)
            f.create_dataset('/baselines', data=tel.baselines)

            # Telescope frequencies
            f.create_dataset('/frequencies', data=tel.frequencies)

            # Write metadata
            f.attrs['beamtransfer_path'] = os.path.abspath(bt.directory)
            f.attrs['ntime'] = ntime

        # Make file marker that the m's have been correctly generated:
        open(completed_file, 'a').close()

        tstream.save()

    mpiutil.barrier()

    return tstream
