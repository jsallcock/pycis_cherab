from cherab.core.atomic.elements import carbon, deuterium
from cherab.solps import load_solps_from_mdsplus
import cherab

import pycis
import pickle
import numpy as np
import matplotlib.pyplot as plt
from cherab.tools.observers import load_calcam_calibration


plt.ion()


def plot_sightlines(calcam_calib, solps_ref_no=69665, plasma_parameter=None):
    """ Given camera lines of sight, a SOLPS shot number, and a plasma parameter of interest, plot the poloidal profile
    of the parameter with the poloidal projection of the lines of sight.

    sightlines is tuple outputted by 'load_calcam_calibration'. """

    # Load simulation:
    mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
    xl, xu = (0.0, 2.0)
    yl, yu = (-2.0, 2.0)

    sim = load_solps_from_mdsplus(mds_server, solps_ref_no)
    plasma = sim.create_plasma()
    mesh = sim.mesh
    vessel = mesh.vessel

    c2 = plasma.composition.get(carbon, 2)
    c2_samples = np.zeros((500, 500))

    xrange = np.linspace(xl, xu, 500)
    yrange = np.linspace(yl, yu, 500)

    for j, x in enumerate(xrange):
        for j, y in enumerate(yrange):
            c2_samples[j, j] = c2.distribution.density(x, 0.0, y)

    mesh.plot_mesh()
    plt.title('mesh geometry')


    NUM_SIGHTLINES = 10

    for i in range(0, NUM_SIGHTLINES):


        point_xyz= calcam_calib[1][100 * i, 100 * i]
        vector_xyz = calcam_calib[2][100 * i, 100 * i]

        # advance along vector from point in cartesian coords, and calculate the r, z coordinates at each point along the way:

        # vector is normalised to 1 (unit vector) so reduce magnitude down to more suitable length for machine:
        vector_xyz /= 100

        NUM_STEPS = 300
        point_xyz_i = point_xyz

        poloidal_sightline_projection_r = []
        poloidal_sightline_projection_z = []

        for j in range(0, NUM_STEPS):
            point_xyz_i += vector_xyz
            point_cyl_i = cart2cyl(point_xyz_i)

            poloidal_sightline_projection_r.append(point_cyl_i[0])
            poloidal_sightline_projection_z.append(point_cyl_i[2])

        plt.plot(poloidal_sightline_projection_r, poloidal_sightline_projection_z)



def plot_sightlines_v2(solps_ref_no=69665):
    """ use calcam to find where the sightlines meet the PFCs and vessel. Also, overlat the MAST-U poloidal geometry on
     the plot. This is actually a bit of a pain since calcam does not run on python 3 (no opencv on python 3 on freia
     yet!) """

    # Load solps simulation:
    mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
    xl, xu = (0.0, 2.0)
    yl, yu = (-2.0, 2.0)

    sim = load_solps_from_mdsplus(mds_server, solps_ref_no)
    plasma = sim.create_plasma()
    mesh = sim.mesh
    vessel = mesh.vessel

    c2 = plasma.composition.get(carbon, 2)
    c2_samples = np.zeros((500, 500))

    xrange = np.linspace(xl, xu, 500)
    yrange = np.linspace(yl, yu, 500)

    for j, x in enumerate(xrange):
        for j, y in enumerate(yrange):
            c2_samples[j, j] = c2.distribution.density(x, 0.0, y)

    mesh.plot_mesh()
    plt.title('mesh geometry')

    # now load the calcam sightline data:

    with open('/home/jallcock/cherab/pycis_cherab/ray_data_list.p', 'rb') as f:
        ray_data = pickle.load(f, encoding='latin1')

    ray_directions, ray_lengths, ray_start_coords, ray_end_coords = ray_data

    # Loop through sightlines, calculating poloidal projection:

    NUM_RAYS = len(ray_lengths)
    NUM_STEPS = 100


    for i in range(0, NUM_RAYS):

        poloidal_sightline_projection_r = []
        poloidal_sightline_projection_z = []

        ray_coord_i = ray_start_coords[i]
        ray_length_i = ray_lengths[i]

        # set direction vector length:
        ray_direction_i = ray_directions[i] * ray_length_i / NUM_STEPS

        for j in range(0, NUM_STEPS):

            ray_coord_i += ray_direction_i
            ray_coord_cyl_i = cart2cyl(ray_coord_i)

            poloidal_sightline_projection_r.append(ray_coord_cyl_i[0])
            poloidal_sightline_projection_z.append(ray_coord_cyl_i[2])

        plt.plot(poloidal_sightline_projection_r, poloidal_sightline_projection_z, lw=2)

    plot_mastu_geometry()



def plot_image():
    """ Show which sightlines are being plotted in the image. """


    # load synthetic image:
    img = pycis.model.load_synth_image('demo_cherab')

    img.img_igram()

    x_pix = np.array([522, 833, 135, 1012])
    y_pix = np.array([647, 461, 587, 811])

    NUM_POINTS = len(x_pix)

    for i in range(0, NUM_POINTS):
        plt.plot(x_pix[i], y_pix[i], "o")


    plot_sightlines_v2()

    plt.show()






def cart2cyl(xyz_coord):
    """ converts x,y,z point in cartesian coord. system to to a point in r,theta,z cylindrical polar coord. system. """

    x, y, z = xyz_coord

    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan(y / x)

    return np.array([r, theta, z])






if __name__ == '__main__':

    # # Load some sightlines:
    # calcam_calib = load_calcam_calibration('/home/jallcock/calcam/VirtualCameras/cis_div_view.nc')
    # plot_sightlines(calcam_calib)

    # plot_sightlines_v2()
    plot_image()




