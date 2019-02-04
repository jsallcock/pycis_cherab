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
    density profiles. Basically a wrapper for cherab, for testing instrument configurations and inversions.

    """

    def __init__(self, inst, dens, dens_sigma, temp, temp_sigma, bfield, name=None):
        """

        :param inst: CIS instrument
        :type inst: pycis.Instrument

        :param dens: [ / m^-3 ]
        :param temp: peak temperature [ eV ]
        :param bfield:

        """

        self.inst = inst
        self.dens = dens
        self.dens_sigma = dens_sigma
        self.temp = temp
        self.temp_sigma = temp_sigma
        self.bfield = bfield

        if name is None:
            name = 'linear_device_test'
        self.name = name
        self.fpath = os.path.join(dpath, name + '.p')

        self.make_plasma()
        self.observe_plasma()
        self.save()


    def make_plasma(self):
        """

        :return:
        """

        ion_density = 8e19
        sigma = 0.25

        # setup scenegraph
        world = World()

        # create atomic data source
        adas = OpenADAS(permit_extrapolation=True)

        # PLASMA
        plasma = Plasma(parent=world)
        plasma.atomic_data = adas
        plasma.geometry = Sphere(sigma * 10.0)
        plasma.geometry_transform = None
        plasma.integrator = NumericalIntegrator(step=sigma / 5.0)

        # define basic distributions
        d_density = pycis_cherab.GaussianBeamProfile(0.5 * ion_density, sigma * 10000)
        e_density = pycis_cherab.GaussianBeamProfile(ion_density, sigma * 10000)
        temperature = 1 + pycis_cherab.GaussianBeamProfile(79, sigma)
        bulk_velocity = ConstantVector3D(Vector3D(0, 0, 0))

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


    def save(self):
        """

        :return:

        """

        with open(self.fpath, 'wb') as f:
            pickle.dump(data, f)

    cis_img = pycis.SynthImage(cis_inst, wl, spec * 1e11)
    cis_img.img_igram()
    plt.show()