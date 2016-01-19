import os
import abc
import numpy as np
from scipy.special import jn

from cora.util import coord
from sphimg.core import telescope
from sphimg.util import config


def ang_conv(ang):
    """
    Covert the string represents of angle in degree to float number in degree.

    Parameters
    ----------
    ang : string
        string represents the angle in the format `xx:xx:xx`
    """
    ang = ang.split(":")
    tmp = 0.0
    for n in range(len(ang)):
        tmp += float(ang[n]) / 60.0**n

    return tmp


def latlon_to_sphpol(latlon):

    zenith = np.array([np.pi / 2.0 - np.radians(latlon[0]),
                       np.remainder(np.radians(latlon[1]), 2*np.pi)])

    return zenith


def beam_circular(angpos, zenith, diameter):
    """Beam pattern for a uniformly illuminated circular dish.

    Parameters
    ----------
    angpos : np.ndarray
        Array of angular positions
    zenith : np.ndarray
        Co-ordinates of the zenith.
    diameter : scalar
        Diameter of the dish (in units of wavelength).

    Returns
    -------
    beam : np.ndarray
        Beam pattern at each position in angpos.
    """

    def jinc(x):
        return 0.5 * (jn(0, x) + jn(2, x))

    x = (1.0 - coord.sph_dot(angpos, zenith)**2)**0.5 * np.pi * diameter

    return 2*jinc(x)


class TlDishArray(config.Reader):
    """A abstract base class describing the Tianlai dishe array for inheriting by sub-classes.

    Attributes
    ----------
    ants : list
        List of antennas to use, number starts from 1.
    zenith : [lat, lon]
        Geometric position of the array.
    pointing: [az, alt, twist]
        Antenna beam to point at (az, alt) with specified right-hand twist to polarizations.
        Polarization y is assumed to be +pi/2 azimuth from pol x.
    freq_lower, freq_higher : scalar
        The lower / upper bound of the lowest / highest frequency bands.
    num_freq : scalar
        The number of frequency bands (only use for setting up the frequency
        binning). Generally using `nfreq` is preferred.
    tsys_flat : scalar
        The system temperature (in K). Override `tsys` for anything more
        sophisticated.
    dish_width : scalar
        Width of the dish in metres.
    center_dish : integer
        The reference dish.
    freq_inds: list
        Choose frequency channels to include.

    """

    __metaclass__ = abc.ABCMeta  # Enforce Abstract class

    # choose antennas to include, number from 1
    ants = config.Property(proptype=list, default=range(1, 17))

    # Set band properties (overriding baseclass)
    # zenith = config.Property(proptype=latlon_to_sphpol, default=[ang_conv('44:9:8.439'), ang_conv('91:48:20.177')])
    zenith = config.Property(proptype=latlon_to_sphpol, default=[ang_conv('44:9:8.439'), ang_conv('0:0:0.0')]) # more convenient for lon = 0
    # Set the antenna beam to point at (az, alt) with specified right-hand twist to polarizations.  Polarization y is assumed to be +pi/2 azimuth from pol x.
    pointing = config.Property(proptype=list, default=[0.0, 90, 0.0]) # [az, alt, twist], Unit: degree
    # freq_lower = config.Property(proptype=float, default=700.0)
    # freq_upper = config.Property(proptype=float, default=800.0)
    # num_freq = config.Property(proptype=int, default=512)
    tsys_flat = config.Property(proptype=float, default=50.0, key='tsys')

    # Properties for the Dish Array
    dish_width = config.Property(proptype=float, default=6.0)
    center_dish = config.Property(proptype=int, default=15)

    freq_inds = config.Property(proptype=list, default=range(512))

    # redefine method to get frequencies
    @property
    def frequencies(self):
        """The centre of each frequency band (in MHz)."""
        return np.load(os.path.dirname(__file__) + '/tldish_freqs.npy')[self.freq_inds]

    # Give the widths in the U and V directions in metres (used for
    # calculating the maximum l and m)
    @property
    def u_width(self):
        return self.dish_width

    @property
    def v_width(self):
        return self.dish_width

    # Set the feed array of feed positions (in metres EW, NS)
    @property
    def _single_feedpositions(self):
        ## An (nfeed,2) array of the feed positions relative to an arbitary point (in m)
        pos = np.loadtxt(os.path.dirname(__file__) + '/16dishes_coord.txt')
        cpos = pos[self.center_dish] # central antenna coordinate
        pos -= cpos
        pos = pos[np.array(self.ants)-1] # choose antennas to include

        return pos


    _point_direction = None

    @property
    def point_dirction(self):
        """The pointing vector [theta, phi], which is the direction of the maximum beam response."""
        if self._point_direction is None:
            self.set_pointing()

        # return self._point_direction
        return np.array([self._point_direction[0], 0.0]) # make phi = 0 for convenience

    def set_pointing(self):
        """Set the antenna beam to point at (az, alt). """

        az, alt, twist = np.radians(self.pointing)
        lat = np.pi/2 - self.zenith[0]
        lon = self.zenith[1]
        saz, caz = np.sin(az), np.cos(az)
        salt, calt = np.sin(alt), np.cos(alt)
        slat, clat = np.sin(lat), np.cos(lat)
        slon, clon = np.sin(lon), np.cos(lon)

        # matrix to convert vector in topocentric coordinate to equatorial coordinate (x starts from the vernal equinox)
        top2eq_m = np.array([[-slon, -slat*clon, clat*clon],
                            [ clon, -slat*slon, clat*slon],
                            [    0,       clat,      slat]])

        p_top = np.array([saz*calt, caz*calt, salt]) # point_direction in topocentric coord
        p_eq = np.dot(top2eq_m, p_top) # point_direction in equatorial coord
        self._point_direction = coord.cart_to_sph(p_eq)[-2:]


class TlUnpolarisedDishArray(TlDishArray, telescope.SimpleUnpolarisedTelescope):
    """A Telescope describing the Tianlai non-polarized dishe array.

    See Also
    --------
    This class also inherits some useful properties, such as `zenith` for
    giving the telescope location and `tsys_flat` for giving the system
    temperature.
    """

    def beam(self, feed, freq):
        """Beam for a particular feed.

        Parameters
        ----------
        feed : integer
            Index for the feed.
        freq : integer
            Index for the frequency.

        Returns
        -------
        beam : np.ndarray
            A Healpix map (of size self._nside) of the beam. Potentially
            complex.
        """
        # return beam_circular(self._angpos, self.zenith,
        #                      self.dish_width / self.wavelengths[freq])
        return beam_circular(self._angpos, self.point_dirction,
                             self.dish_width / self.wavelengths[freq])


class TlPolarisedDishArray(telescope.SimplePolarisedTelescope):
    """A Telescope describing the Tianlai polarized dishe array.

    See Also
    --------
    This class also inherits some useful properties, such as `zenith` for
    giving the telescope location and `tsys_flat` for giving the system
    temperature.
    """

    # Implement the X and Y beam patterns (assuming all feeds are identical).
    # These need to return a vector for each position on the sky
    # (self._angpos) in thetahat, phihat coordinates.
    def beamx(self, feed, freq):
        """Beam for the X polarisation feed.

        Parameters
        ----------
        feed : integer
            Index for the feed.
        freq : integer
            Index for the frequency.

        Returns
        -------
        beam : np.ndarray
            Healpix maps (of size [self._nside, 2]) of the field pattern in the
            theta and phi directions.
        """
        # Calculate beam amplitude
        # beam = beam_circular(self._angpos, self.zenith,
        #                      self.dish_width / self.wavelengths[freq])
        return beam_circular(self._angpos, self.point_dirction,
                             self.dish_width / self.wavelengths[freq])

        # Add a vector direction to beam - X beam is EW (phihat)
        beam = beam[:, np.newaxis] * np.array([0.0, 1.0])

        return beam

    def beamy(self, feed, freq):
        """Beam for the Y polarisation feed.

        Parameters
        ----------
        feed : integer
            Index for the feed.
        freq : integer
            Index for the frequency.

        Returns
        -------
        beam : np.ndarray
            Healpix maps (of size [self._nside, 2]) of the field pattern in the
            theta and phi directions.
        """
        # Calculate beam amplitude
        # beam = beam_circular(self._angpos, self.zenith,
        #                      self.dish_width / self.wavelengths[freq])
        return beam_circular(self._angpos, self.point_dirction,
                             self.dish_width / self.wavelengths[freq])

        # Add a vector direction to beam - Y beam is NS (thetahat)
        # Fine provided beam does not cross a pole.
        beam = beam[:, np.newaxis] * np.array([1.0, 0.0])

        return beam
