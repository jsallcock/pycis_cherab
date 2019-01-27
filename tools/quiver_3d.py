from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

import pycis
import pycis_cherab
import scipy.integrate
import os
import sys

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

#####

mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
solps_ref_number = 69566

world = World()

# Load all parts of mesh with chosen material
MESH_PARTS = ['/projects/cadmesh/mast/mastu-light/mug_centrecolumn_endplates.stl',
            '/projects/cadmesh/mast/mastu-light/mug_divertor_nose_plates.stl']

for path in MESH_PARTS:
    import_stl(path, parent=world, material=AbsorbingSurface())  # Mesh with perfect absorber

# Load plasma from SOLPS model
sim = load_solps_from_mdsplus(mds_server, solps_ref_number)
plasma = sim.create_plasma(parent=world)
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
mesh = sim.mesh
vessel = mesh.vessel

# get the species distributions
electrons = plasma.electron_distribution
d0 = plasma.composition.get(deuterium, 0)
d1 = plasma.composition.get(deuterium, 1)
c2 = plasma.composition.get(carbon, 2)


def quiver2d_pan():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Make the grid
    delta = 0.0333
    x_lo, x_hi = (-2., 2.)
    y_lo, y_hi = (-2., 2.)

    xs, ys = (np.arange(x_lo, x_hi, delta), np.arange(y_lo, y_hi, delta))

    for x, y in product(xs, ys):

        bx, by, bz = plasma.b_field(x, y, 0)
        vx, vy, vz = c2.distribution.bulk_velocity(x, y, 0) / 1.e5

        # print(x, y, z)
        # print(bx, by, bz)
        # print('----')

        if bx != 0.:
            ax.quiver(x, y, bx, by, width=0.004, headwidth=5, norm=True, color='k')
            ax.quiver(x, y, vx, vy, width=0.004, headwidth=5, norm=False, color='r')

    plt.show()

    return

def quiver2d_pol():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Make the grid
    delta = 0.05
    x_lo, x_hi = (-2., 2.)
    z_lo, z_hi = (-2., 2.)

    xs, zs = (np.arange(x_lo, x_hi, delta), np.arange(z_lo, z_hi, delta))

    for x, z in product(xs, zs):

        bx, by, bz = plasma.b_field(x, 0, z)
        vx, vy, vz = c2.distribution.bulk_velocity(x, 0, z) / 1.e5

        # print(x, y, z)
        # print(bx, by, bz)
        # print('----')

        if bx != 0.:
            ax.quiver(x, z, bx, bz, width=0.004, headwidth=5, norm=True, color='k')
            ax.quiver(x, z, vx, vz, width=0.004, headwidth=5, norm=False, color='r')

    plt.show()

    return


def quiver3d():
    fig3d = plt.figure()
    ax = fig3d.gca(projection='3d')

    # Make the grid
    delta = 0.1
    x_lo, x_hi = (-2., 2.)
    y_lo, y_hi = (-2., 2.)
    z_lo, z_hi = (-2., 2.)

    xs, ys, zs = (np.arange(x_lo, x_hi, delta), np.arange(y_lo, y_hi, delta), np.arange(z_lo, z_hi, delta))

    for x, y, z in product(xs, ys, zs):

        bx, by, bz = plasma.b_field(x, y, z)
        # vx, vy, vz = c2.distribution.bulk_velocity(x, y, z) / 1.e5

        # print(x, y, z)
        # print(bx, by, bz)
        # print('----')

        if bx != 0.:
            ax.quiver(x, y, z, bx, by, bz, normalize=False, lw=1, length=0.1)

    plt.show()

    return

if __name__ == '__main__':
    quiver2d_pol()


