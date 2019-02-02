cimport cython
from libc.math cimport exp
from cherab.core.math.function cimport Function3D


cdef class GaussianBeamProfile(Function3D):
    """
    modelled after cherab's GaussianVolume example Function3D, a function for Gaussian beam profiles (like a linear
    plasma device)
    """

    cdef double peak
    cdef double sigma
    cdef double _constant
    cdef bint _cache
    cdef double _cache_x, _cache_y, _cache_z, _cache_v

    def __init__(self, peak, sigma):
        self.peak = peak
        self.sigma = sigma
        self._constant = (2*self.sigma*self.sigma)

        # last value cache
        self._cache = False
        self._cache_x = 0.0
        self._cache_y = 0.0
        self._cache_z = 0.0
        self._cache_v = 0.0

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef double v

        if self._cache:
            if x == self._cache_x and y == self._cache_y and z == self._cache_z:
                return self._cache_v

        v = self.peak * exp(-(y*y + z*z) / self._constant)
        self._cache = True
        self._cache_x = x
        self._cache_y = y
        self._cache_z = z
        self._cache_v = v
        return v



cdef class GaussianVolume(Function3D):

    cdef double peak
    cdef double sigma
    cdef double _constant
    cdef bint _cache
    cdef double _cache_x, _cache_y, _cache_z, _cache_v

    def __init__(self, peak, sigma):
        self.peak = peak
        self.sigma = sigma
        self._constant = (2*self.sigma*self.sigma)

        # last value cache
        self._cache = False
        self._cache_x = 0.0
        self._cache_y = 0.0
        self._cache_z = 0.0
        self._cache_v = 0.0

    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef double v

        if self._cache:
            if x == self._cache_x and y == self._cache_y and z == self._cache_z:
                return self._cache_v

        v = self.peak * exp(-(x*x + y*y + z*z) / self._constant)
        self._cache = True
        self._cache_x = x
        self._cache_y = y
        self._cache_z = z
        self._cache_v = v
        return v