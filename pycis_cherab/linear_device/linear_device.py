import pycis
import pycis_cherab

# external imports
import matplotlib.pyplot as plt
import os
import pickle
from scipy.constants import electron_mass, atomic_mass

# cherab imports
from cherab.core.math.function.vectorfunction3d import PythonVectorFunction3D
from cherab.core.distribution import Maxwellian
import numpy as np
from cherab.core.model import ExcitationLine, RecombinationLine, Bremsstrahlung
from cherab.core.math import Constant3D, ConstantVector3D

# Cherab and raysect imports
from cherab.core import Species, Maxwellian, Plasma, Line, elements
from cherab.openadas import OpenADAS

# Core and external imports
from raysect.optical import World, translate, rotate, Vector3D, Point3D, Ray
from raysect.optical.observer.pipeline.spectral import SpectralPowerPipeline2D, SpectralRadiancePipeline2D
from raysect.optical.observer.pipeline import RGBPipeline2D, PowerPipeline2D
from raysect.primitive import Sphere, Cylinder
from raysect.optical.observer import PinholeCamera
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

dpath = '/Users/jsallcock/Documents/physics/phd/py_repo/pycis_cherab/pycis_cherab/linear_device/saved_data'


def load_linear_device(fname):
    """
    load previously saved instance of LinearDevice

    :return:

    """

    fpath = os.path.join(dpath, fname + '.p')
    if os.path.isfile(fpath):
        with open(fpath, 'rb') as f:
            linear_device = pickle.load(f)
    else:
        raise Exception('please enter valid fname.')

    return linear_device


class LinearDevice:
    """
    Synthetic CIS measurements on a linear plasma device. Plasma beam assumed to have Gaussian temperature and
    density profiles. Basically a wrapper for cherab, for testing instrument configurations, spectral lineshape models
    and inversions in a simple geometry.

    """

    def __init__(self, inst, plasma_params, line, sim_name=None):
        """
        :param inst: CIS instrument
        :type inst: pycis.Instrument

        :param plasma_params: dict of plasma parameters. keys:
        -'dens_peak': peak density [ / m^-3 ] (assumes ion density = electron density)
        -'dens_sigma': density profile width [ m ]
        -'temp_peak': peak temperature [ eV ] (assumes ion temperature = electron temperature)
        -'temp_sigma': temperature profile width [ m ]
        -'bfield': uniform magnetic field strength along linear device axis [ T ]
        -'plasma_len': length of [ m ]

        :param line: string corresponding to observed Balmer line

        :param sim_name: simulation name for saving
        :type sim_name: str

        """

        # TODO explicitly define coordinate system
        # TODO instrument placement
        # TODO sightline figure plotting
        # TODO recreate Magnum PSI approximate plasma profiles

        self.inst = inst

        self.dens_peak = plasma_params['dens_peak']
        self.dens_sigma = plasma_params['dens_sigma']
        self.temp_peak = plasma_params['temp_peak']
        self.temp_sigma = plasma_params['temp_sigma']
        self.bfield = plasma_params['bfield']
        self.plasma_len = plasma_params['plasma_len']

        self.line = line
        self.sim_name = sim_name

        self.make_plasma()
        self.observe_plasma()
        self.save()


    def make_plasma(self):
        """

        :return:
        """

        # setup scenegraph
        world = World()

        # create atomic data source
        adas = OpenADAS(permit_extrapolation=True)

        plasma = Plasma(parent=world)
        plasma.atomic_data = adas
        plasma.geometry = Cylinder(self.dens_sigma * 8, self.plasma_len)
        plasma.geometry_transform = None
        plasma.integrator = NumericalIntegrator(step=self.dens_sigma / 20)

        # define basic distributions
        d_density = pycis_cherab.GaussianBeamProfile(self.dens_peak, self.dens_sigma)
        e_density = pycis_cherab.GaussianBeamProfile(self.dens_peak, self.dens_sigma)
        temperature = pycis_cherab.GaussianBeamProfile(self.temp_peak, self.temp_sigma)

        bulk_velocity = ConstantVector3D(Vector3D(0, 0, 0))  # hard-coded to zero bulk flow for now
        d_distribution = Maxwellian(d_density, temperature, bulk_velocity,
                                    elements.deuterium.atomic_weight * atomic_mass)
        e_distribution = Maxwellian(e_density, temperature, bulk_velocity, electron_mass)

        d0_species = Species(elements.deuterium, 0, d_distribution)
        d1_species = Species(elements.deuterium, 1, d_distribution)

        # define species
        plasma.b_field = ConstantVector3D(Vector3D(1.0, 1.0, 1.0))
        plasma.electron_distribution = e_distribution
        plasma.composition = [d0_species, d1_species]

        # Setup elements.deuterium lines
        d_alpha = Line(elements.deuterium, 0, (3, 2))
        d_beta = Line(elements.deuterium, 0, (4, 2))
        d_gamma = Line(elements.deuterium, 0, (5, 2))
        d_delta = Line(elements.deuterium, 0, (6, 2))
        d_epsilon = Line(elements.deuterium, 0, (7, 2))

        plasma.models = [
            # Bremsstrahlung(),
            # ExcitationLine(d_alpha),
            # ExcitationLine(d_beta),
            ExcitationLine(d_gamma),
            # ExcitationLine(d_delta),
            # ExcitationLine(d_epsilon),
            # RecombinationLine(d_alpha),
            # RecombinationLine(d_beta),
            RecombinationLine(d_gamma),
            # RecombinationLine(d_delta),
            # RecombinationLine(d_epsilon)
        ]

        # alternate geometry
        # plasma.geometry = Cylinder(sigma * 2.0, sigma * 10.0)
        # plasma.geometry_transform = translate(0, -sigma * 5.0, 0) * rotate(0, 90, 0)

        return

    def observe_plasma(self):
        """

        :return:
        """

        plt.ion()

        # define pipelines
        spectral = SpectralRadiancePipeline2D()
        rgb = RGBPipeline2D(display_unsaturated_fraction=0.8, name="sRGB")
        # Get the power and raw spectral data for scientific use.
        power_unfiltered = PowerPipeline2D(display_unsaturated_fraction=0.8, name="Unfiltered Power (W)")
        power_unfiltered.display_update_time = 15

        camera = PinholeCamera(sensor_dim, pipelines=[spectral, rgb, power_unfiltered], fov=35, parent=world,
                               transform=translate(0, 0, -3.5))
        # camera.render_engine = SerialEngine()
        camera.spectral_rays = 1
        camera.spectral_bins = 65
        camera.pixel_samples = 20
        camera.max_wavelength = 434.5
        camera.min_wavelength = 433.5

        plt.ion()
        camera.observe()

        plt.ioff()
        plt.show(block=True)

        spec = np.moveaxis(spectral.frame.mean, -1, 0)
        spec = np.rot90(spec, axes=(-2, -1))
        wl = spectral.wavelengths * 1e-9

        return

    def plot_view(self):
        """

        :return:
        """


        return


    def save(self):
        """

        :return:

        """

        if self.sim_name is None:
            self.sim_name = 'linear_device_test'
        fpath = os.path.join(dpath, self.sim_name + '.p')

        with open(fpath, 'wb') as f:
            pickle.dump(data, f)