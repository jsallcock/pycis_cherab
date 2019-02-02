#!/usr/bin/env python
# some code from Tom for suppressing matplotlib outputs when running batch scripts:

# import os, logging
import pickle, inspect
# batch_mode = os.getenv('LOADL_ACTIVE', None)
# print(batch_mode)
# job_name = os.getenv('LOADL_JOB_NAME', None)
# # execution_mode = os.getenv('LOADL_STEP_TYPE', None)
# # logger = logging.getLogger(__name__)
# # logger.setLevel(logging.DEBUG)
# # logger.info('Executing run_elzar.py, batch_mode={}, job_name={}'.format(batch_mode, job_name))
#
# if batch_mode == 'yes':
#
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#
#
#     print('In batch mode')
# else:
#     import matplotlib
#     import matplotlib.pyplot as plt
print('this is a print statement')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Core and external imports
import pycis
import pycis_cherab
import numpy as np
from scipy.io.netcdf import netcdf_file
plt.ion()

from raysect.core import Vector3D, Point3D
from raysect.core.ray import Ray as CoreRay
from raysect.core.workflow import SerialEngine
from raysect.optical import World
from raysect.optical.observer import VectorCamera
from raysect.primitive.mesh import import_stl
from raysect.optical.material.lambert import Lambert
from raysect.optical.material.absorber import AbsorbingSurface
from cherab.core.model.lineshape import GaussianLine, LineShapeModel, StarkBroadenedLine
from raysect.optical.observer import RGBPipeline2D, SpectralPowerPipeline2D, PowerPipeline2D

# Cherab and raysect imports
from cherab.core.atomic import Line
from cherab.core.atomic.elements import deuterium, carbon
from cherab.core.model import ExcitationLine, RecombinationLine
# from cherab.tools.observers import load_calcam_calibration
# from cherab.solps import load_solps_from_mdsplus
from cherab.openadas import OpenADAS

mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
world = World()

# Load all parts of mesh with chosen material -- TODO move this part somewhere else
MESH_PARTS = ['/projects/cadmesh/mast/mastu-light/mug_centrecolumn_endplates.stl',
              '/projects/cadmesh/mast/mastu-light/mug_divertor_nose_plates.stl']

# for path in MESH_PARTS:
#     import_stl(path, parent=world, material=AbsorbingSurface())  # Mesh with perfect absorber


# save_img_path = '/Users/GraceYoung/Documents/joe/code/pycis_cherab/saved_cherab_outputs/'

import os

from raysect.optical.material import AbsorbingSurface, Lambert


def make_cis_synth_image(solps_ref_no, line_name, cam_calib_name, cherab_image_name=None):
    """ generate / load an instance of CherabImageMASTU

    this may be moved elsewhere."""

    # load calibration synth image:
    # calib_image = pycis.model.load_synth_image('demo_cherab')

    cherab_image_mastu = load_cherab_image_mastu(solps_ref_no, line_name, cam_calib_name, cherab_image_name=cherab_image_name)

    # convert wavelength [nm] --> [m]
    cherab_image_mastu.wavelengths = (cherab_image_mastu.wavelengths * 1e-9).astype(np.float64)

    # take absolute intensity (need to ask Matt Carr about negative intensities!)
    cherab_image_mastu.spec_cube = (abs(cherab_image_mastu.spec_cube) * 4e3).astype(np.float64)

    # make SynthSpectraCherab
    camera_pycis = pycis.model.load_component('photron_SA4', type='camera')
    instrument = pycis.model.load_component('mastu_ciii_cherab', type='instrument')

    image_spectra = pycis.model.SpectraCherab(cherab_image_mastu.wavelengths, cherab_image_mastu.spec_cube, instrument, 'mastu_ciii_cherab')
    synth_image = pycis.model.SynthImageCherab(instrument, image_spectra, 'mastu_ciii_cherab_div')
    synth_image.save()

    return


def load_cherab_image_mastu(solps_ref_no, line_name, cam_calib_name, cherab_image_name=None):
    if cherab_image_name is None:
        cherab_image_name = str(solps_ref_no) + '_' + cam_calib_name + '_' + line_name

    cherab_image_mastu = pickle.load(open(os.path.join(pycis_cherab.save_img_path, cherab_image_name + '.p'), 'rb'))

    return cherab_image_mastu


class CherabImageMASTU(object):
    """
    base class for generating CIS images of MAST-U

    """

    def __init__(self, solps_ref_no, line_name, cam_calib_name, wavelengths=None, spec_cube=None, cherab_image_name=None):
        """

        :param solps_ref_no:
        :param line_name:
        :param cam_calib_name:
        :param wavelengths:
        :param spec_cube:
        :param cherab_image_name:
        """

        self.solps_ref_no = solps_ref_no

        self.line_name = line_name
        self.cam_calib_name = cam_calib_name

        if cherab_image_name is None:
            self.name = str(solps_ref_no) + '_' + cam_calib_name + '_' + line_name
        else:
            self.name = cherab_image_name

        if wavelengths is not None and spec_cube is not None:
            self.wavelengths = wavelengths
            self.spec_cube = spec_cube
        else:
            # generate and save data
            self.wavelengths, self.spec_cube = self.make()

        self.save()

    def make(self):
        """ Run cherab to calculate the image spectra. """

        # Load all parts of mesh with chosen material
        MESH_PARTS = ['/projects/cadmesh/mast/mastu-light/mug_centrecolumn_endplates.stl',
                      '/projects/cadmesh/mast/mastu-light/mug_divertor_nose_plates.stl']

        for path in MESH_PARTS:
            import_stl(path, parent=world, material=AbsorbingSurface())  # Mesh with perfect absorber
            # import_stl(path, parent=world, material=Lambert(ConstantSF(0.25)))  # Mesh with 25% Lambertian reflectance
            # import_stl(path, parent=world, material=Debug(Vector3D(0.0, 1.0, 0.0)))  # Mesh with debugging material

        # import_mastu_mesh(world)
        sim = load_solps_from_mdsplus(mds_server, self.solps_ref_no)
        plasma = sim.create_plasma(parent=world)
        plasma.atomic_data = OpenADAS(permit_extrapolation=True)
        mesh = sim.mesh
        vessel = mesh.vessel

        # Pick emission models
        # d_alpha = Line(deuterium, 0, (3, 2))
        # plasma.models = [ExcitationLine(d_alpha), RecombinationLine(d_alpha)]

        if self.line_name == 'D_gamma':
            d_gamma = Line(deuterium, 0, (5, 2))
            plasma.models = [ExcitationLine(d_gamma, lineshape=StarkBroadenedLine), RecombinationLine(d_gamma, lineshape=StarkBroadenedLine)]
            min_wavelength, max_wavelength = 433.85, 434.15
        elif self.line_name == 'C III':
            ciii_465 = Line(carbon, 2, ('2s1 3p1 3P4.0', '2s1 3s1 3S1.0'))
            plasma.models = [ExcitationLine(ciii_465)]
            min_wavelength, max_wavelength = 464.65, 465.35
        else:
            raise Exception()

        # Select from available Cameras
        # camera_config = load_calcam_calibration('/home/mcarr/mastu/cameras/mug_bulletb_midplane.nc')
        # camera_config = load_calcam_calibration('/home/mcarr/mastu/cameras/mug_divcam_isp.nc')
        camera_config = load_calcam_calibration(os.path.join(pycis_cherab.save_calib_path, self.cam_calib_name + '.nc'))

        # RGB pipeline for visualisation
        rgb = RGBPipeline2D(display_unsaturated_fraction=0.96, name="sRGB")

        # Get the power and raw spectral data for scientific use.
        power_unfiltered = PowerPipeline2D(display_unsaturated_fraction=0.96, name="Unfiltered Power (W)")
        power_unfiltered.display_update_time = 15
        spectral = SpectralPowerPipeline2D()

        # Setup camera for interactive use...
        pixels_shape, pixel_origins, pixel_directions = camera_config
        pixel_origins = np.flipud(np.rot90(pixel_origins, k=0))
        pixel_directions = np.flipud(np.rot90(pixel_directions, k=0))

        camera_cherab = VectorCamera(pixel_origins, pixel_directions, pipelines=[rgb, power_unfiltered, spectral], parent=world)
        camera_cherab.min_wavelength = min_wavelength
        camera_cherab.max_wavelength = max_wavelength

        camera_cherab.spectral_bins = 30
        camera_cherab.pixel_samples = 1
        # camera_cherab.quiet = True
        # camera_cherab.display_progress = False

        camera_cherab.observe()

        # OR Setup camera for batch run on cluster
        # pixels_shape, pixel_origins, pixel_directions = camera_config
        # pixel_origins = np.flipud(np.rot90(pixel_origins, k=-1))
        # pixel_directions = np.flipud(np.rot90(pixel_directions, k=-1))
        #
        # camera = VectorCamera(pixel_origins, pixel_directions, pipelines=[spectral], parent=world)
        # camera.spectral_samples = 15
        # camera.pixel_samples = 50
        # camera.display_progress = False
        # camera.accumulate = True

        # # start ray tracing
        # for p in range(1, 5000):
        #     print("Rendering pass {} ({} samples/pixel)..."
        #           "".format(p, camera.accumulated_samples + camera.pixel_samples * camera.spectral_rays))
        #     camera.observe()
        #     camera.save("mastu_divcam_dalpha_{}_samples.png".format(camera.accumulated_samples))
        #     print()

        return spectral.wavelengths, spectral.frame.mean

    def imshow(self, ax, pixel_idxs=None):
        """ Quick, image one spectral bin to see what the image looks like.

        :param pixel_idxs: tuple of pixel coordinates to plot like ([x1, y1], [x2, y2], ...)
        """

        im = ax.pcolormesh(np.sum(self.spec_cube, axis=-1))
        cbar = plt.colorbar(im, ax=ax)

        if pixel_idxs is not None:
            for pixel_idx in pixel_idxs:
                ax.scatter(pixel_idx[0], pixel_idx[1], marker='o')
        return

    def plot_ray_param(self, ax, pixel_idxs, param):
        """

        :param ax:
        :param pixel_idxs: list of pixel coordinate tuples
        :param param:
        :return:
        """

        start_points, forward_vectors = self.load_calcam_calibration_partial(pixel_idxs)

        solps_profile = pycis_cherab.SolpsProfile(self.solps_ref_no)

        for start_point, forward_vector in zip(start_points, forward_vectors):
            solps_profile.plot_ray_param(ax, start_point, forward_vector, param)

    def plot_ray_path(self, ax, pixel_idxs):

        start_points, forward_vectors = self.load_calcam_calibration_partial(pixel_idxs)

        solps_profile = pycis_cherab.SolpsProfile(self.solps_ref_no)

        for start_point, forward_vector in zip(start_points, forward_vectors):
            solps_profile.plot_ray_path(ax, start_point, forward_vector)

    def plot_param_profile(self, ax, param):
        """

        :param ax:
        :param param:
        :return:
        """

        solps_profile = pycis_cherab.SolpsProfile(self.solps_ref_no)

        solps_profile.plot_param_profile(ax, param)

        return

    def plot_ray_lineshape(self, ax, pixel_idxs, line_name, **kwargs):

        start_points, forward_vectors = self.load_calcam_calibration_partial(pixel_idxs)
        solps_profile = pycis_cherab.SolpsProfile(self.solps_ref_no)

        for start_point, forward_vector in zip(start_points, forward_vectors):
            solps_profile.plot_ray_lineshape(ax, start_point, forward_vector, line_name, **kwargs)

    def get_calcam_load_path(self):
        return os.path.join(pycis_cherab.save_calib_path, self.cam_calib_name + '.nc')

    def plot_mastu_camera_view(self, ax, **kwargs):
        """ Given a calcam calibration, plot the sightline extrema over the MAST-U poloidal profile of a specified
        plasma parameter."""

        ray_xs = np.arange(100, 1000, 100)
        ray_ys = np.arange(100, 1000, 100)

        pixel_idxs = []
        for ray_y in ray_ys:
            for ray_x in ray_xs:
                pixel_idxs.append((ray_x, ray_y))

        # manually select (roughly) the two 'extrema' rays in the R-Z plane (TODO better way to do this?)
        pixel_idxs = [(100, 450), (1000, 575), (500, 380)]

        start_points, forward_vectors = self.load_calcam_calibration_partial(pixel_idxs)

        plasma = self.load_solps_plasma_object(solps_ref_no=self.solps_ref_no)

        for ray_idx, (start_point, forward_vector) in enumerate(zip(start_points, forward_vectors)):

            ray_param = self.get_ray_param(start_point, forward_vector, plasma)
            ax.plot(ray_param['r'], ray_param['z'], **kwargs)

        self.imshow(pixel_idxs)

        return

    def load_calcam_calibration_partial(self, pixel_idxs):
        """
        cherab.tools.observers.load_calcam_calibration -- But only load a few user-specified pixels to save time.

        :param calcam_load_path:
        :param pixel_idxs: tuple of pixel coordinates like ([x1, y1], [x2, y2], ...)
        :return:
        """
        calcam_load_path = self.get_calcam_load_path()
        camera_config = netcdf_file(calcam_load_path)

        ray_start_coords = camera_config.variables['RayStartCoords'].data
        ray_end_coords = camera_config.variables['RayEndCoords'].data

        ray_start_coords = np.flipud(np.rot90(ray_start_coords, k=2))
        ray_end_coords = np.flipud(np.rot90(ray_end_coords, k=2))

        # print(ray_start_coords)

        output_shape = len(pixel_idxs)
        pixel_origins = np.empty(shape=output_shape, dtype=np.dtype(object))
        pixel_directions = np.empty(shape=output_shape, dtype=np.dtype(object))

        for i, pixel_idx in enumerate(pixel_idxs):
            print(i, pixel_idx)
            xi, yi, zi = ray_start_coords[pixel_idx[0], pixel_idx[1], :]
            xj, yj, zj = ray_end_coords[pixel_idx[0], pixel_idx[1], :]

            pixel_origins[i] = Point3D(xi, yi, zi)
            pixel_directions[i] = Vector3D(xj - xi, yj - yi, zj - zi).normalise()

        return pixel_origins, pixel_directions

    def save(self):
        pickle.dump(self, open(os.path.join(pycis_cherab.save_img_path, self.name + '.p'), 'wb'))


if __name__ == '__main__':
    from pycis_cherab import CherabImageMASTU

    solps_ref_no = 69636
    line_name = 'D_gamma'
    cam_calib_name = 'mastu_div_cis'

    cim = CherabImageMASTU(solps_ref_no, line_name, cam_calib_name)
    #
    # cim = pycis_cherab.load_cherab_image_mastu(solps_ref_no, line_name, cam_calib_name)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    pixel_idxs = [(100, 450), (1000, 575), (500, 380)]
    param = 'D_gamma_total_emiss'

    cim.plot_ray_param(ax1, pixel_idxs, param)
    cim.imshow(ax2, pixel_idxs=pixel_idxs)

    cim.plot_param_profile(ax3, param)
    cim.plot_ray_path(ax3, pixel_idxs)

    cim.plot_ray_lineshape(ax4, pixel_idxs, 'D_gamma_recom')

    plt.show()


    #
    # cim.plot_param_profile(ax1, solps_ref_no_manual=69636, param='ne')
    # pycis_cherab.tools.plot_mastu_poloidal_geometry(ax1, color='dimgrey', lw=2)
    # cim.plot_mastu_camera_view(ax1)
    #
    # ax1.set_aspect('equal')
    #
    # fig2, ax2 = plt.subplots()
    # pixel_idxs = [(100, 450), (1000, 575), (500, 380)]
    #
    # cim.plot_ray_param(ax2, pixel_idxs, 'd0v_dot_l')
    #
    # cim.get_lineshape(pixel_idxs)
    #
    # plt.show()


    # si = make_cis_synth_image(solps_ref_no, line_name, cam_calib_name)
    # make_cis_synth_image(solps_ref_no, line_name, cam_calib_name, cherab_image_name=str(solps_ref_no)+'_' + cam_calib_name + '_reflections_180_samples_per_pix')
    # si.img_igram_intensity()
    # plt.show()





