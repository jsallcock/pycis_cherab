import calcam
import calcam.raytrace
import numpy as np
import matplotlib.pyplot as plt
import pickle


def calcam_sightlines(calcam_calib, calcam_cad, savepath, x_pix=None, y_pix=None):
    """ ***** MUST BE RUN IN PYTHON 2 *****

    Given a calcam calibration object and  calcam cad model object, saves the ray directions and lengths.

    This a wrapper for calcam's raycast_pixels, which runs in python 2 but will output the data in a python 3 friendly
    format."""

    ray_caster = calcam.raytrace.RayCaster(calcam_calib, calcam_cad)
    ray_data = ray_caster.raycast_pixels(x=x_pix, y=y_pix)

    ray_directions = ray_data.get_ray_directions()
    ray_lengths = ray_data.get_ray_lengths()
    ray_start_coords = ray_data.ray_start_coords
    ray_end_coords = ray_data.ray_end_coords

    # now output as a pickle

    ray_data_list = [ray_directions, ray_lengths, ray_start_coords, ray_end_coords]

    pickle.dump(ray_data_list, open(savepath, 'wb'))




if __name__ == '__main__':

    calcam_calib = calcam.VirtualCalib('cis_div_view')
    calcam_cad = calcam.machine_geometry.MAST_U()
    savepath = '/home/jallcock/cherab/pycis_cherab/HL10_ray_data.p'

    # x_pix = np.array([522, 833, 135, 1012])
    # y_pix = np.array([647, 461, 587, 811])

    calcam_sightlines(calcam_calib, calcam_cad, savepath)




