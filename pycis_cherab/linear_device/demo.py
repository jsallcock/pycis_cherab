import pycis
import pycis_cherab
import numpy as np

"""

demo for the LinearDevice class, used to simulate observation of a linear plasma with a Gaussian density and 
temperature profile.

"""

simulation_name = 'demo_1'

# define CIS instrument

# camera
bit_depth = 16
sensor_dim = (2560, 2160)
pix_size = 6.5e-6
qe = 0.6
epercount = 0.46  # [ e / count ]
cam_noise = 2.5  # [ e ]

# instrument rear lens
flength = 85e-3  # [ m ]
back_lens = pycis.Lens(flength)
cam = pycis.Camera(bit_depth, sensor_dim, pix_size, qe, epercount, cam_noise)


# interferometer
pol_1 = pycis.LinearPolariser(0.)
sp_1 = pycis.SavartPlate(np.pi / 4, 4.0e-3)
wp_1 = pycis.UniaxialCrystal(np.pi / 4, 4.48e-3, 0)
pol_2 = pycis.LinearPolariser(0.)
interferometer = [pol_1, sp_1, wp_1, pol_2]

inst = pycis.Instrument(cam, back_lens, interferometer)

# define instrument position and orientation

# field of view angle
fov = 35

# define the plasma parameters and profiles
