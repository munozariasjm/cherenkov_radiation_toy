import numpy as np
from scipy.constants import fine_structure, speed_of_light, elementary_charge
import matplotlib.pyplot as plt
from typing import List
from scipy import integrate
from .schemas import Particle, Medium, alpha, c, z
import matplotlib
from matplotlib import rc
import matplotlib.pylab as plt

from matplotlib import rc
import matplotlib.pylab as plt

try:
    rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 18})
    rc("text", usetex=True)
except Exception as e:
    print(e)


class CherenkovPerParticle:
    """Simulate Cherenkov radiation for a single particle in a medium."""

    def __init__(self, particle, medium):
        self.particle = particle
        self.medium = medium

    def cherenkov_angle(self, omega):
        """Calculate the Cherenkov angle for a given frequency."""
        return np.arccos(1 / (self.particle.beta * self.medium.refractive_index(omega)))

    def cherenkov_angle_range(self, omega_range):
        """Calculate the Cherenkov angle for a range of frequencies."""
        return np.array([self.cherenkov_angle(omega) for omega in omega_range])

    def cherenkov_angle_range_degrees(self, omega_range):
        """Calculate the Cherenkov angle for a range of frequencies in degrees."""
        return np.degrees(self.cherenkov_angle_range(omega_range))

    def frank_tamm_dw_dx(self, omega, particle):
        """Calculate the Frank-Tamm formula for the Cherenkov radiation.
        Returns the energy radiated per unit length per unit frequency.
        """
        fact = (
            (particle.charge**2 / (4 * np.pi))
            * self.medium.permeability(omega)
            * omega
        )
        return fact * (
            1 - 1 / (self.particle.beta**2 * self.medium.refractive_index(omega) ** 2)
        )

    def frank_tamm_dx(self, min_omega, max_omega):
        """Calculate the Frank-Tamm formula for the Cherenkov radiation.
        Returns the energy radiated per unit length.
        """
        return integrate.quad(
            lambda omega: self.frank_tamm_dw_dx(omega, self.particle),
            min_omega,
            max_omega,
        )[0]

    def cherenkov_angle_range_degrees_plot(self, omega_range):
        """Plot the Cherenkov angle for a range of frequencies in degrees."""
        plt.plot(omega_range, self.cherenkov_angle_range_degrees(omega_range))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Cherenkov angle (degrees)")
        plt.show()


class CherenkovMontecarlo:
    def __init__(
        self, medium, speed_distribution: np.array, n: int = 1_000_000
    ) -> None:
        self.medium = medium
        self.speed_distribution = speed_distribution
        self.particles = self.create_particles(n)
        self.cherenkovs = self._cherenkovs(self.particles)

    def create_particles(self, n: int) -> List[Particle]:
        """Create a list of particles with the given speed distribution."""
        return [
            Particle(speed, 1, 1)
            for speed in np.random.choice(self.speed_distribution, int(n))
        ]

    def _cherenkovs(self, particles) -> np.array:
        return [CherenkovPerParticle(particle, self.medium) for particle in particles]

    def plot_speed_distribution(self, n_bins: int = int(1e3)) -> None:
        """Plot the speed distribution."""
        plt.hist([s.get_speed() for s in self.particles], bins=n_bins)
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Number of particles")
        plt.show()

    def get_angles_distribution(self, omega) -> np.array:
        """Get the distribution of Cherenkov angles."""
        return np.array(
            [cherenkov.cherenkov_angle(omega) for cherenkov in self.cherenkovs]
        )

    def plot_angles_distribution(self, omega, n_bins, degrees=False) -> None:
        """Plot the distribution of Cherenkov angles."""
        angles = self.get_angles_distribution(omega)
        label = "Cherenkov angle (rad)"
        if degrees:
            angles = np.degrees(angles)
            label = "Cherenkov angle (degrees)"
        plt.hist(angles, bins=n_bins)
        plt.xlabel(label)
        plt.ylabel("Number of particles")
        plt.show()

    def get_energy_per_length(self, min_omega, max_omega) -> np.array:
        """Get the energy per length for a given frequency."""
        return np.array(
            [
                cherenkov.frank_tamm_dx(min_omega, max_omega)
                for cherenkov in self.cherenkovs
            ]
        )
