import numpy as np
from scipy.constants import fine_structure, speed_of_light, elementary_charge
import matplotlib.pyplot as plt
from typing import List, Union, Callable
from scipy import integrate


# Define constants
alpha = fine_structure
c = speed_of_light
z = 1  # for an electron

class Particle:
    """Class for a particle with mass, charge and speed"""
    def __init__(self, speed: float, mass: float, charge: float) -> None:
        self.mass = mass
        self.charge = charge
        self.v = speed
        self.gamma = 1 / np.sqrt(1 - self.v**2 / c**2)
        self.beta = self.v / c

class Medium:
    """Class for a medium with refractive index and permeability"""
    def __init__(self, refractive_index:  Union[float, callable],
                 permeability_index: Union[float, callable]=1):
        self.n = refractive_index
        self.mu = permeability_index

    def refractive_index(self, omega: Union[float, callable]) -> float:
        """ The refractive index of the medium

        Args:
            omega (Union[float, callable]): Angular frequency of the light
        """
        try:
            return self.n(omega)
        except TypeError:
            return self.n

    def permeability(self, omega: Union[float, callable]):
        """ The permeability of the medium

        Args:
            omega (Union[float, callable]): Angular frequency of the light
        """
        try:
            return self.mu(omega)
        except TypeError:
            return self.mu
