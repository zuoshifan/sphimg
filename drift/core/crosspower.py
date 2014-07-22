import numpy as np

from drift.util import mpiutil
from drift.util import npcov
from drift.core import psmc


class CrossPower(psmc.PSMonteCarlo):

    crosspower = True

    def _work_fisher_bias_m(self, mi):
        """Worker routine for calculating the Fisher and bias for a given m.

        This method estimates both quantities using Monte-Carlo estimation,
        and the fact that Cov(q_a, q_b) = F_ab.

        Parameters
        ----------
        mi : integer
            m-mode to calculate.

        Returns
        -------
        fisher : np.ndarray[nbands, nbands]
            Fisher matrix.
        bias : np.ndarray[nbands]
            Bias vector.
        """

        qa = np.zeros((self.nbands + 1, self.nsamples), dtype=np.complex128)

        # Split calculation into subranges to save on memory usage
        num, starts, ends = mpiutil.split_m(self.nsamples, (self.nsamples / 1000) + 1)

        for n, s, e in zip(num, starts, ends):

            x1 = self.gen_sample(mi, n)
            x2 = self.gen_sample(mi, n)
            qa[:, s:e] = self.q_estimator(mi, x1, x2, noise=True)

        ft = npcov.cov(qa)

        fisher = ft[:self.nbands, :self.nbands]
        bias = ft[-1, :self.nbands]

        return fisher, bias
