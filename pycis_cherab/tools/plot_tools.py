import numpy as np
import matplotlib.pyplot as plt
import os, glob, sys, time
import pycis_cherab
from scipy.io.netcdf import netcdf_file

start_cherab_import = time.time()
# Core and external imports
from raysect.core import Vector3D, Point3D
from raysect.core.ray import Ray as CoreRay
from raysect.primitive.mesh import Mesh, import_stl
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.optical.observer import FibreOptic, SpectralPowerPipeline0D
from raysect.optical import World, translate, rotate, rotate_basis, Spectrum

# Cherab and raysect imports
from cherab.core.atomic import Line
from cherab.core.atomic.elements import deuterium, carbon
from cherab.openadas import OpenADAS
from cherab.solps import load_solps_from_mdsplus
from cherab.core.model import ExcitationLine, RecombinationLine, Bremsstrahlung
from cherab.core.model.lineshape import StarkBroadenedLine
from cherab.tools.observers import load_calcam_calibration
end_cherab_import = time.time()
print('imports', end_cherab_import - start_cherab_import)

mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
world = World()

# Load all parts of mesh with chosen material
MESH_PARTS = ['/projects/cadmesh/mast/mastu-light/mug_centrecolumn_endplates.stl',
              '/projects/cadmesh/mast/mastu-light/mug_divertor_nose_plates.stl']

for path in MESH_PARTS:
    import_stl(path, parent=world, material=AbsorbingSurface())  # Mesh with perfect absorber


def plot_mastu_poloidal_geometry(ax, **kwargs):
    """ plot the poloidal projection of MAST-U tile surfaces on a given matplotlib axis object."""

    r = [2.000, 2.000, 1.191, 0.893, 0.868, 0.847, 0.832, 0.823, 0.820, 0.820, 0.825, 0.840, 0.864, 0.893, 0.925,
         1.690, 2.000, 2.000, 1.319, 1.769, 1.730, 1.350, 1.090, 0.900, 0.360, 0.333, 0.333, 0.261, 0.261, 0.261, 0.333,
         0.333, 0.360, 0.900, 1.090, 1.350, 1.730, 1.769, 1.319, 2.000, 2.000, 1.690, 0.925, 0.893, 0.864, 0.840, 0.825,
         0.820, 0.820, 0.823, 0.832, 0.847, 0.868, 0.893, 1.191, 2.000, 2.000]

    z = [-0.000, -1.007, -1.007, -1.304, -1.334, -1.368, -1.404, -1.442, -1.481, -1.490, -1.522, -1.551, -1.573, -1.587,
         -1.590, -1.552, -1.560, -2.169, -2.169, -1.719, -1.680, -2.060, -2.060, -1.870, -1.330, -1.303, -1.100, -0.500,
         0.000, 0.500, 1.100, 1.303, 1.330, 1.870, 2.060, 2.060, 1.680, 1.719, 2.169, 2.169, 1.560, 1.552, 1.590, 1.587,
         1.573, 1.551, 1.522, 1.490, 1.481, 1.442, 1.404, 1.368, 1.334, 1.304, 1.007, 1.007, 0.000]

    r = np.array(r)
    z = np.array(z)

    ax.plot(r, z, ** kwargs)

    return


# if __name__ == '__main__':
#
#     # calcam_load_path = os.path.join(pycis_cherab.save_calib_path,  'mastu_div_cis' + '.nc')
#     #
#     # pixel_origins, pixel_directions = load_calcam_calibration_partial(calcam_load_path, [(500, 500)])
#
#
#     # fig, ax = plt.subplots()
#     # plot_mastu_poloidal_geometry(ax)
#     # plot_mastu_camera_sightlines(ax, 'mastu_div_cis')
#     # ax.set_aspect('equal', 'datalim')
#     #
#     # plt.show()

