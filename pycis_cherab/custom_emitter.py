"""

following the simple Gaussian volume and balmer series demo from cherab.core

"""
import pycis
import pycis_cherab

# external imports
import matplotlib.pyplot as plt
import os
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

# file details
saved_spectra_path = '/Users/jsallcock/Documents/physics/phd/py_repo/pycis_cherab/pycis_cherab/saved_spectra'
saved_spectra_name = 'd_gamma_gaussian_beam_test_1'
spec_f = os.path.join(saved_spectra_path, saved_spectra_name + '.npy')
wavelength_f = os.path.join(saved_spectra_path, saved_spectra_name + '_wavelength.npy')
overwrite = True

# define CIS instrument
bit_depth = 16
sensor_dim = (1024, 1024)
pix_size = 6.5e-6
qe = 0.6
epercount = 0.46  # [e / count]
cam_noise = 2.5
cam = pycis.Camera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)
flength = 85e-3
backlens = pycis.Lens(flength)

# interferometer components
pol_1 = pycis.LinearPolariser(0.)
sp_1 = pycis.SavartPlate(np.pi / 4, 4.0e-3)
wp_1 = pycis.UniaxialCrystal(np.pi / 4, 4.48e-3, 0)
pol_2 = pycis.LinearPolariser(0.)
interferometer = [pol_1, sp_1, wp_1, pol_2]
cis_inst = pycis.Instrument(cam, backlens, interferometer)

if os.path.isfile(spec_f) and os.path.isfile(wavelength_f) and overwrite is False:
    spec = np.load(spec_f)
    wl = np.load(wavelength_f)

else:

    # tunables
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
    d_density = pycis_cherab.GaussianBeamProfile(0.5 * ion_density, sigma*10000)
    e_density = pycis_cherab.GaussianBeamProfile(ion_density, sigma*10000)
    temperature = 1 + pycis_cherab.GaussianBeamProfile(79, sigma)
    bulk_velocity = ConstantVector3D(Vector3D(0, 0, 0))

    d_distribution = Maxwellian(d_density, temperature, bulk_velocity, elements.deuterium.atomic_weight * atomic_mass)
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
    wl = spectral.wavelengths * 1e9

    np.save(spec_f, spec)
    np.save(wavelength_f, wl)

cis_img = pycis.SynthImage(cis_inst, wl, spec)
cis_img.img_igram()
plt.show()


