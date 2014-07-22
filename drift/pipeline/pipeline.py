import os.path

import yaml

from drift.util import mpiutil
from drift.core import manager
from drift.util import config
from drift.util import typeutil
from drift.pipeline import timestream


def fixpath(path):
    """Fix up path (expanding variables etc.)"""
    path = os.path.expanduser(path)
    path = os.path.expandvars(path)
    path = os.path.normpath(path)

    return path


class PipelineManager(config.Reader):
    """Manage and run the pipeline.

    Attributes
    ----------
    timestream_directory : string
        Directory that the timestream is stored in.
    product_directory : string
        Directory that the analysis products are stored in.
    output_directory : string
        Directory to store timestream outputs in.

    generate_mmodes : boolean
        Calculate m-modes.
    generate_svdmodes : boolean
        Calculate svd-modes.
    generate_klmodes : boolean
        Calculate KL-modes.
    generate_powerspectra : boolean
        Estimate powerspectra.

    klmodes : list
        List of KL-filters to apply ['klname1', 'klname2', ...]
    powerspectra : list
        List of powerspectra to apply. Requires entries to be dicts
        like [ { 'psname' : 'ps1', 'klname' : 'dk'}, ...]

    fullmap_fwhm : scalar
        The full width half max parameter of the Gaussian symmetric beam function in radians to smooth alms before `mapmake_full`.
    svdmap_fwhm : scalar
        The full width half max parameter of the Gaussian symmetric beam function in radians to smooth alms before `mapmake_svd`.
    """

    # Directories
    product_directory = config.Property(proptype=str, default='')

    # Actions to perform
    generate_mmodes = config.Property(proptype=bool, default=True)
    generate_svdmodes = config.Property(proptype=bool, default=True)
    generate_klmodes = config.Property(proptype=bool, default=True)
    generate_powerspectra = config.Property(proptype=bool, default=True)
    generate_crosspower = config.Property(proptype=bool, default=True)
    generate_full_map = config.Property(proptype=bool, default=True)
    generate_svd_map = config.Property(proptype=bool, default=True)
    generate_kl_map = config.Property(proptype=bool, default=True)

    no_m_zero = config.Property(proptype=bool, default=True)

    # Specific products to use.
    klmodes = config.Property(proptype=list, default=[])
    powerspectra = config.Property(proptype=list, default=[])
    klmaps = config.Property(proptype=list, default=[])
    crosspower = []

    # Specific map-making options
    fullmap_fwhm = config.Property(proptype=typeutil.nonnegative_float, default=0.0)
    svdmap_fwhm = config.Property(proptype=typeutil.nonnegative_float, default=0.0)
    full_rank_ratio = config.Property(proptype=typeutil.nonnegative_float, default=0.0)
    svd_rank_ratio = config.Property(proptype=typeutil.nonnegative_float, default=0.0)
    kl_rank_ratio = config.Property(proptype=typeutil.nonnegative_float, default=0.0)
    full_lcut = config.Property(proptype=typeutil.none_or_natural_int, default=None)
    svd_lcut = config.Property(proptype=typeutil.none_or_natural_int, default=None)
    kl_lcut = config.Property(proptype=typeutil.none_or_natural_int, default=None)
    nside = config.Property(proptype=typeutil.power_of_2, default=128)
    wiener = config.Property(proptype=bool, default=False)

    timestreams = {}
    simulations = {}
    prodmanager = None

    collect_klmodes = config.Property(proptype=bool, default=True)


    @classmethod
    def from_configfile(cls, configfile):

        c = cls()
        c.load_configfile(configfile)

        return c


    def load_configfile(self, configfile):


        with open(configfile, 'r') as f:
            yconf = yaml.safe_load(f)

        ## Global configuration
        ## Create output directory and copy over params file.
        if 'config' not in yconf:
            raise Exception('Configuration file must have an \'config\' section.')

        # Load config in from file.
        self.read_config(yconf['config'])

        # Load in timestream information
        if 'timestreams' not in yconf:
            raise Exception('Configuration file must have an \'timestream\' section.')

        for tsconf in yconf['timestreams']:

            name = tsconf['name']
            tsdir = fixpath(tsconf['directory'])

            # Load ProductManager and Timestream
            if self.prodmanager is None:
                self.prodmanager = manager.ProductManager.from_config(self.product_directory)
            ts = timestream.Timestream(tsdir, name, self.prodmanager)

            if 'output_directory' in tsconf:
                outdir = fixpath(tsconf['output_directory'])
                ts.output_directory = outdir

            ts.no_m_zero = self.no_m_zero

            self.timestreams[name] = ts

            if 'simulate' in tsconf:
                self.simulations[name] = tsconf['simulate']

        if 'crosspower' in yconf:

            self.crosspower = [ xp for xp in yconf['crosspower'] ]



    def simulate(self):

        for tsname, simconf in self.simulations.items():

            if mpiutil.rank0:
                print
                print '=' * 80
                print 'Start simulation for %s...' % tsname

            ts = self.timestreams[tsname]

            # if os.path.exists(ts._ffile(0)):
            #     print "Looks like timestream already exists. Skipping...."
            # else:
                # m = manager.ProductManager.from_config(simconf['product_directory'])
                # timestream.simulate(m, ts.directory, **simconf)
            timestream.simulate(self.prodmanager, ts.directory, tsname, **simconf)



    def generate(self):
        """Generate pipeline outputs."""

        for tsname, tsobj in self.timestreams.items():

            if self.generate_mmodes:
                if mpiutil.rank0:
                    print
                    print '=' * 80
                    print "Generating m-modes (%s)..." % tsname

                tsobj.generate_mmodes()


            if self.generate_full_map:
                if mpiutil.rank0:
                    print
                    print '=' * 80
                    print "Generating full map (%s)..." % tsname

                tsobj.mapmake_full(self.nside, 'map_full.hdf5', self.fullmap_fwhm, rank_ratio=self.full_rank_ratio, lcut=self.full_lcut)


            if self.generate_svdmodes:
                if mpiutil.rank0:
                    print
                    print '=' * 80
                    print "Generating svd modes (%s)..." % tsname

                tsobj.generate_mmodes_svd()


            if self.generate_svd_map:
                if mpiutil.rank0:
                    print
                    print '=' * 80
                    print "Generating SVD map (%s)..." % tsname

                tsobj.mapmake_svd(self.nside, 'map_svd.hdf5', self.svdmap_fwhm, rank_ratio=self.svd_rank_ratio, lcut=self.svd_lcut)


            if self.generate_klmodes:
                for klname in self.klmodes:
                    if mpiutil.rank0:
                        print
                        print '=' * 80
                        print "Generating KL filter (%s:%s)..." % (tsname, klname)

                    tsobj.set_kltransform(klname)
                    tsobj.generate_mmodes_kl()

                    if self.collect_klmodes:
                        tsobj.collect_mmodes_kl()


            if self.generate_kl_map:
                for klname in self.klmaps:
                    if mpiutil.rank0:
                        print
                        print '=' * 80
                        print "Generating KL map (%s:%s)..." % (tsname, klname)

                    tsobj.set_kltransform(klname)

                    mapfile = 'map_%s.hdf5' % klname
                    tsobj.mapmake_kl(self.nside, mapfile, wiener=self.wiener, rank_ratio=self.kl_rank_ratio, lcut=self.kl_lcut)


            if self.generate_powerspectra:
                for ps in self.powerspectra:

                    psname = ps['psname']
                    klname = ps['klname']

                    if mpiutil.rank0:
                        print
                        print '=' * 80
                        print "Estimating powerspectra (%s:%s)..." % (tsname, psname)

                    tsobj.set_kltransform(klname)
                    tsobj.set_psestimator(psname)

                    tsobj.powerspectrum()



        if self.generate_crosspower:
            for xp in self.crosspower:

                psname = xp['psname']
                klname = xp['klname']

                tslist = []

                if mpiutil.rank0:
                    print
                    print '=' * 80
                    print 'Estimating cross powerspectra %s...' % psname

                for tsname in xp['timestreams']:

                    tsobj = self.timestreams[tsname]
                    tsobj.set_kltransform(klname)
                    # tsobj.set_psestimator(psname) # not need for `psname` is an argument given to cross_powerspectum

                    tslist.append(tsobj)

                psfile = os.path.abspath(os.path.expandvars(os.path.expanduser(xp['psfile'])))

                timestream.cross_powerspectrum(tslist, psname, psfile)



        if mpiutil.rank0:
            print
            print
            print "========================================"
            print "=                                      ="
            print "=           DONE AT LAST!!             ="
            print "=                                      ="
            print "========================================"


    run = generate
