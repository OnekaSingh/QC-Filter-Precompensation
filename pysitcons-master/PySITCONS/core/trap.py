from operator import pos
import numpy as np
from numba.experimental import jitclass
import numba as nb
from .verlet import verlet
from .spl import spline

spec = [
    ("ions", nb.float64[:, :]),
    ("trap", nb.float64[:, :, :]),
    ("splcoeff", nb.float64[:, :, :]),
    ("volt", nb.float64[:, :]),
    ("qdm", nb.float64),
    ("interaction", nb.boolean),
]


@jitclass(spec)
class red_system(verlet, spline):
    def __init__(self, trap, interaction=True):

        self.splcoeff = np.empty(trap.shape)
        self.volt = np.zeros((len(trap), 1))

        for i in range(len(trap)):
            self.splcoeff[i, 0] = self.splrep(trap[i, 0], trap[i, 1])
            self.splcoeff[i, 1] = self.splrep(trap[i, 0], trap[i, 2])

        self.trap = trap
        self.interaction = interaction

        q = 1.60217662e-19
        m = 1.66054e-27 * 39.96259
        self.qdm = q / m

    def propagate(self, ramp, endtime, dt):
        """Evolve a given system through time.

        Parameters
        ----------
        ramp : np.array
            must have size 1 + number_of_segements,
            first dimension being the time

        endtime : float
            time to integrate up to

        dt : float
            time bewteen to integration steps
        
        """
        n = int(endtime / dt)
        tpoints = np.linspace(0, endtime, n)

        self.volt = np.zeros((len(ramp)-1 , n))
        for i in range(len(ramp) - 1):
            self.volt[i] = np.interp(tpoints, ramp[0], ramp[i + 1])

        return self.integrate(endtime, dt)

    def setIons(self, positions):
        """set the initial positions of the ions.
        
        Parameters
        ----------
        posistions : np.array
            has size (#numberOfIons, positions).
            For one dim (x, 3), for 3 dim (x, 3)
            with position, velocity, acceleration
        """
        self.ions = positions

    def getAcc(self, position=None, k=0):
        if position is None:
            position = self.ions[:, 0]

        acc = np.zeros(position.shape)

        for i in range(len(self.trap)):
            acc += (self.volt[i, k] * 
            self.splev(position, 
            self.trap[i, 0], 
            self.trap[i, 2], 
            self.splcoeff[i, 1]))

        if not self.interaction:
            return acc * self.qdm

        ion_acc = np.zeros(len(position))
        for i in range(len(position)):
            for l in range(len(position)):
                if i == l:
                    continue

                ion_acc[i] += (
                    1.60217662 ** 2
                    / (np.square(position[i] - position[l]) * 1.66054 * 39.96259 * 40 * np.pi * 8.8541878128)
                    * np.sign(position[i] - position[l])
                )

        return acc * self.qdm + ion_acc

    def getDAcc(self, position=None, k=0):
        if position is None:
            position = self.ions[:, 0]

        acc = np.zeros(position.shape)

        for i in range(len(self.trap)):
            acc += self.volt[i, k] * self.splevd(position, self.trap[i, 0], self.trap[i, 2], self.splcoeff[i, 1])

        if not self.interaction:
            return acc * self.qdm

        ion_acc = np.zeros(len(position))
        for i in range(len(position)):
            for l in range(len(position)):
                if i == l:
                    continue

                ion_acc[i] += (
                    -(1.60217662 ** 2)
                    / ((position[i] - position[l]) ** 3 * 1.66054 * 39.96259 * 40 * np.pi * 8.8541878128)
                ) * 2
                # TODO signum?

        return acc * self.qdm + ion_acc

    def getPot(self, position=None, k=0):
        """
        returns the potential at givin positions

        Parameters
        ----------
        position : np.array
            positions where to evaluate the potential
        """
        if position is None:
            position = self.ions[:, 0]

        pot = np.zeros(position.shape)

        for i in range(len(self.trap)):
            pot += self.volt[i, k] * self.splev(position, self.trap[i, 0], self.trap[i, 1], self.splcoeff[i, 0])

        if not self.interaction:
            return pot

        return pot

    def getPotcurv(self, position, k=0):
        acc = np.zeros(position.shape)

        for i in range(len(self.trap)):
            acc += self.volt[i, k] * self.splevd(position, self.trap[i, 0], self.trap[i, 2], self.splcoeff[i, 1])

        return -acc * self.qdm
