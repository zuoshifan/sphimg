import numpy as np

from cylsim import cylinder, util


def gaussian_fwhm(x, fwhm):

    sigma = fwhm / (8.0*np.log(2.0))**0.5
    x2 = x**2 / (2*sigma**2)
    
    return np.exp(-x2)


class RestrictedCylinder(cylinder.UnpolarisedCylinderTelescope):

    beam_height = 30.0
    beam_type = 'box'

    __config_table_ =   {
                          'beam_height'   : [float, 'beam_height'],
                          'beam_type'     : [str,   'beam_type']
                        }


    def __init__(self, *args, **kwargs):
        super(RestrictedCylinder, self).__init__(*args, **kwargs)

        self.add_config(self.__config_table_)


    #@util.cache_last
    def beam_gaussian(self, feed, freq):

        pointing = self.zenith
        bdist = (self._angpos - pointing[np.newaxis, :])
        bdist = np.abs(np.where((bdist[:, 1] < np.pi)[:, np.newaxis], bdist, bdist - np.array([0, 2*np.pi])[np.newaxis, :]))

        fwhm_x = 1.0 / (self.cylinder_width / self.wavelengths[freq])

        beam = np.sinc(bdist[:, 1] / fwhm_x) * gaussian_fwhm(bdist[:, 0], np.radians(self.beam_height))

        return beam


    #@util.cache_last
    # def beam_sinc(self, feed, freq):

    #     pointing = self.zenith
    #     bdist = (self._angpos - pointing[np.newaxis, :])
    #     bdist = np.abs(np.where((bdist[:, 1] < np.pi)[:, np.newaxis], bdist, bdist - np.array([0, 2*np.pi])[np.newaxis, :]))

    #     D_x = (self.cylinder_width / self.wavelengths[freq])
    #     D_y = (self.feed_spacing / self.wavelengths[freq])

    #     beam = np.sinc(bdist[:, 0] * D_y) * np.sinc(bdist[:, 1] * D_x)

    #     return beam

    #@util.cache_last
    def beam_box(self, feed, freq):

        pointing = self.zenith
        bdist = (self._angpos - pointing[np.newaxis, :])
        bdist = np.abs(np.where((bdist[:, 1] < np.pi)[:, np.newaxis], bdist, bdist - np.array([0, 2*np.pi])[np.newaxis, :]))

        D_x = (self.cylinder_width / self.wavelengths[freq])

        beam = np.sinc(bdist[:, 1] * D_x) * (np.abs(bdist[:, 0] / np.radians(self.beam_height)) < 0.5)

        return beam


    def beam(self, *args, **kwargs):
        bdict = {
                  'gaussian' : self.beam_gaussian,
                  #'sinc'     : self.beam_sinc,
                  'box'      : self.beam_box
                }

        return bdict[self.beam_type](*args, **kwargs)

