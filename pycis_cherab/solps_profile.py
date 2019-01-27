import numpy as np
import matplotlib.pyplot as plt

# raysect imports
from raysect.optical import World
from raysect.primitive.mesh import import_stl
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.core import Vector3D, Point3D
from raysect.core.ray import Ray as CoreRay
from raysect.optical import World, translate, rotate, rotate_basis, Spectrum

# cherab imports
from cherab.solps import load_solps_from_mdsplus
from cherab.core.atomic import Line
from cherab.core.atomic.elements import deuterium, carbon
from cherab.openadas import OpenADAS
from cherab.core.model import ExcitationLine, RecombinationLine

# my imports
import pycis_cherab
import pycis
import pystark

mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
world = World()

# define some parameters and defaults for plotting
xl, xu = (0.0, 2.0)
yl, yu = (-2.2, 2.2)


class SolpsProfile(object):
    def __init__(self, solps_ref_no):
        """ trace rays through a SOLPS plasma profile, plot a SOLPS poloidal profile, etc. """

        sim = load_solps_from_mdsplus(mds_server, solps_ref_no)
        self.plasma = sim.create_plasma(parent=world)
        self.plasma.atomic_data = OpenADAS(permit_extrapolation=True)

        self.load_mastu_mesh()

        # define some commonly used spectral lines

        # Balmer series

        d_beta = Line(deuterium, 0, (4, 2))
        d_beta_excit = ExcitationLine(d_beta)
        d_beta_recom = RecombinationLine(d_beta)

        d_gamma = Line(deuterium, 0, (5, 2))
        d_gamma_excit = ExcitationLine(d_gamma)
        d_gamma_recom = RecombinationLine(d_gamma)

        d_delta = Line(deuterium, 0, (6, 2))
        d_delta_excit = ExcitationLine(d_delta)
        d_delta_recom = RecombinationLine(d_delta)

        # Impurities

        # # C III at 465 nm
        # ciii = Line(carbon, 2, ('2s1 3p1 3P4.0', '2s1 3s1 3S1.0'))
        # ciii_excit = ExcitationLine(ciii)
        # ciii_recom = RecombinationLine(ciii)

        self.line_names = ['D_beta', 'D_gamma', 'D_delta']

        self.lines = {'D_beta_excit': d_beta_excit, 'D_beta_recom': d_beta_recom,
                      'D_gamma_excit': d_gamma_excit, 'D_gamma_recom': d_gamma_recom,
                      'D_delta_excit': d_delta_excit, 'D_delta_recom': d_delta_recom}

        self.lines_n_upper = {'D_beta_excit': 4, 'D_beta_recom': 4,
                      'D_gamma_excit': 5, 'D_gamma_recom': 5,
                      'D_delta_excit': 6, 'D_delta_recom': 6}

        self.plasma.models = [d_beta_excit, d_beta_recom,
                              d_gamma_excit, d_gamma_recom,
                              d_delta_excit, d_delta_recom]

        # define the functions used to get parameter values

        self.valid_param_get_fns = {'ne': self.plasma.electron_distribution.density,
                                    'ni': self.plasma.composition.get(deuterium, 0).distribution.density,
                                    'te': self.plasma.electron_distribution.effective_temperature,
                                    'ti': self.plasma.composition.get(deuterium, 0).distribution.effective_temperature,

                                    'D_beta_excit_emiss': self.get_emissivity_fn(line_name='D_beta', line_type='excit'),
                                    'D_beta_recom_emiss': self.get_emissivity_fn(line_name='D_beta', line_type='recom'),
                                    'D_beta_total_emiss': self.get_emissivity_fn(line_name='D_beta', line_type='total'),

                                    'D_gamma_excit_emiss': self.get_emissivity_fn(line_name='D_gamma', line_type='excit'),
                                    'D_gamma_recom_emiss': self.get_emissivity_fn(line_name='D_gamma', line_type='recom'),
                                    'D_gamma_total_emiss': self.get_emissivity_fn(line_name='D_gamma', line_type='total'),

                                    'D_delta_excit_emiss': self.get_emissivity_fn(line_name='D_delta', line_type='excit'),
                                    'D_delta_recom_emiss': self.get_emissivity_fn(line_name='D_delta', line_type='recom'),
                                    'D_delta_total_emiss': self.get_emissivity_fn(line_name='D_delta', line_type='total')
                                    }

        self.valid_params = list(self.valid_param_get_fns.keys())

    def get_emissivity_fn(self, line_name, line_type):
        """ Only works for Balmer series so far. """

        # assert line_name in list(self.lines.keys())
        assert line_name in self.line_names
        assert line_type in ['excit', 'recom', 'total']

        if line_type != 'total':

            def emissivity_fn(x, y, z):

                line = self.lines[line_name + '_' + line_type]
                sample_point = Point3D(x, y, z)
                direction = Vector3D(1, 1, 1)  # ie. we don't care about view direction for now.

                emiss = line.emission(sample_point, direction, Spectrum(380, 700, 1000)).total()

                return emiss

        else:

            def emissivity_fn(x, y, z):

                line_excit = self.lines[line_name + '_excit']
                line_recom = self.lines[line_name + '_recom']

                sample_point = Point3D(x, y, z)
                direction = Vector3D(1, 1, 1)  # ie. we don't care about view direction for now.

                excit_emiss = line_excit.emission(sample_point, direction, Spectrum(380, 700, 1000)).total()
                recom_emiss = line_recom.emission(sample_point, direction, Spectrum(380, 700, 1000)).total()

                return excit_emiss + recom_emiss

        return emissivity_fn

    @staticmethod
    def load_mastu_mesh(mode='basic'):
        """ load the MAST-U CAD mesh into world """

        valid_modes = ['basic', 'full']
        assert mode in valid_modes

        if mode == 'basic':
            MESH_PARTS = ['/projects/cadmesh/mast/mastu-light/mug_centrecolumn_endplates.stl',
                          '/projects/cadmesh/mast/mastu-light/mug_divertor_nose_plates.stl']

            for path in MESH_PARTS:
                import_stl(path, parent=world, material=AbsorbingSurface())  # Mesh with perfect absorber

        elif mode == 'full':
            pycis_cherab.import_mastu_mesh(world)

    def get_ray_params(self, start_point, forward_vector, params):
        """
        returns parameters of interest along the given pixel's

        :param start_point:
        :param forward_vector:
        :param params: list of strings corresponding to the parameters needed. MUST BE LIST.
        :return:
        """

        forward_vector = forward_vector.normalise()  # ensure normalised

        # for some reason i get hitpoint logged at r  = 1.702 m, this is a hack to get around this.
        hack_increment = 0.1
        r = np.sqrt(start_point[0] ** 2 + start_point[1] ** 2)
        start_point_hack = start_point
        while r > 1.702:
            start_point_hack += hack_increment * forward_vector
            r = np.sqrt(start_point_hack[0] ** 2 + start_point_hack[1] ** 2)

        intersection = world.hit(CoreRay(start_point_hack, forward_vector))
        if intersection is not None:
            hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
            print("Intersection with the vessel was found at ", hit_point)
            xh, yh, zh = hit_point
            rh = np.sqrt(xh ** 2 + yh ** 2)
            print('where: r = ', rh)
        else:
            hit_point = start_point + 1e-3 * forward_vector
        # print('intersection:', end_intersection - start_intersection)

        # trace along the ray's path
        parametric_vector = start_point.vector_to(hit_point)
        npts = 1000  # number of samples along ray path
        t_samples = np.arange(0, 1, 1 / npts)

        ray_params = {}
        param_get_fns = {}
        for param in params:
            assert (param in self.valid_params)  # check input parameter names valid
            ray_params.update({param: np.zeros(npts)})  # array to populate with parameter
            param_get_fns.update({param: self.valid_param_get_fns[param]})  # function to return the parameter

        ray_x, ray_y, ray_z, ray_r, ray_dist = [], [], [], [], []

        for ray_path_idx, t in enumerate(t_samples):
            # print(i, '/', len(t_samples))

            # Get new sample point location and log distance
            x = start_point.x + parametric_vector.x * t
            y = start_point.y + parametric_vector.y * t
            z = start_point.z + parametric_vector.z * t
            sample_point = Point3D(x, y, z)

            ray_x.append(x)
            ray_y.append(y)
            ray_r.append(np.sqrt(x ** 2 + y ** 2))
            ray_z.append(z)
            ray_dist.append(start_point.distance_to(sample_point))

            for param in params:
                ray_params[param][ray_path_idx] = param_get_fns[param](x, y, z)

        ray_params.update({'x': np.array(ray_x)})  # array to populate with parameter
        ray_params.update({'y': np.array(ray_y)})  # array to populate with parameter
        ray_params.update({'z': np.array(ray_z)})  # array to populate with parameter
        ray_params.update({'r': np.array(ray_r)})  # array to populate with parameter
        ray_params.update({'dist': np.array(ray_dist)})  # array to populate with parameter

        return ray_params

    def plot_ray_param(self, ax, start_point, forward_vector, param, **kwargs):
        """
        plot the selected parameter along the line of sight of the selected ray

        :param ax:
        :param start_point: Point3D object
        :param forward_vector: Vector3D object
        :param param: must be str, corresponding to parameter needed
        :param kwargs: for plotting
        :return:
        """

        ray_params = self.get_ray_params(start_point, forward_vector, [param])
        ax.plot(ray_params['dist'], ray_params[param], **kwargs)
        ax.set_title(param)

        return

    def plot_ray_path(self, ax, start_point, forward_vector, **kwargs):
        """
        plot ray on r-Z poloidal plane.

        :param ax:
        :param start_point:
        :param forward_vector:
        :param kwargs:
        :return:
        """

        ray_params = self.get_ray_params(start_point, forward_vector, [])
        ax.plot(ray_params['r'], ray_params['z'], **kwargs)

    def get_param_profiles(self, params):
        """ load the solps poloidal profile data for selected parameters.

        It may be necessary to do this by bulk / save the result in the future.

        :param params: list of parameters whose profiles will be returned
        :return:
        """

        n_samples_x = 500
        n_samples_y = 500
        xrange = np.linspace(xl, xu, n_samples_x)
        yrange = np.linspace(yl, yu, n_samples_y)

        param_profiles = {}
        param_get_fns = {}
        for param in params:
            assert (param in self.valid_params)  # check input parameter names valid
            param_profiles.update({param: np.zeros((n_samples_y, n_samples_x))})  # array to populate with parameter
            param_get_fns.update({param: self.valid_param_get_fns[param]})  # function to return the parameter

        for i, x in enumerate(xrange):
            for j, y in enumerate(yrange):
                for param in params:
                    param_profiles[param][j, i] = param_get_fns[param](x, 0.0, y)

        return param_profiles

    def plot_param_profile(self, ax, param):
        """

        :param ax:
        :param param: must be string
        :return:
        """

        # load profile data
        param_profile = self.get_param_profiles([param])[param]

        # make plot
        im = ax.imshow(param_profile, extent=[xl, xu, yl, yu], origin='lower')
        cbar = plt.colorbar(im, ax=ax)
        ax.set_xlim(xl, xu)
        ax.set_ylim(yl, yu)
        ax.set_title(param)

        pycis_cherab.plot_mastu_poloidal_geometry(ax, color='white', lw=2)

        return

    def get_ray_lineshape(self, start_point, forward_vector, line_name):
        """

        :param start_point:
        :param forward_vector:
        :param line_names: MUST BE LIST -- TODO change this, its really clumsy
        :return:
        """

        params = ['ne', 'te']
        assert line_name in list(self.lines.keys())
        params.append(line_name + '_emiss')

        ray_params = self.get_ray_params(start_point, forward_vector, params)

        ray_ne = ray_params['ne']
        ray_te = ray_params['te']
        ray_line_emiss = ray_params[line_name + '_emiss']
        n_upper = self.lines_n_upper[line_name]

        # get a reasonable wavelength axis
        wls = pystark.get_wavelength_axis(n_upper, np.max(ray_ne), np.max(ray_te), 0, no_fwhm=5, npts=300)
        ls = np.zeros_like(wls)

        for ne, te, line_emiss in zip(ray_ne, ray_te, ray_line_emiss):

            if ne > 0 and te > 0 and line_emiss > 0:
                bls = pystark.BalmerLineshape(n_upper, ne, te, 0.0, line_model='stehle param', wl_axis=wls)

                ls += line_emiss * bls.ls_szd

        ls /= np.sum(ray_line_emiss)

        return ls, wls

    def plot_ray_lineshape(self, ax, start_point, forward_vector, line_name, **kwargs):

        ls, wls = self.get_ray_lineshape(start_point, forward_vector, line_name)
        ax.plot(wls, ls, **kwargs)

        return







if __name__ == '__main__':

    sp = SolpsProfile(69636)

    print('okay, we got this far')


    # fig2, ax2 = plt.subplots()
    # fig3, ax3 = plt.subplots()
    #
    # sp.plot_param_profile(ax1, 'D_beta_emiss_excit')
    # sp.plot_param_profile(ax2, 'D_gamma_emiss_excit')
    # sp.plot_param_profile(ax3, 'D_delta_emiss_excit')
    #
    # plt.show()






