from __future__ import division, print_function, absolute_import
import math
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage import _nd_image

def rotate_scipy(input, angle, axes=(1, 0), reshape=True,
           output=None, order=3,
           mode='constant', cval=0.0, prefilter=True):
    """
    Rotate an array.

    The array is rotated in the plane defined by the two axes given by the
    `axes` parameter using spline interpolation of the requested order.

    Parameters
    ----------
    input : ndarray
        The input array.
    angle : float
        The rotation angle in degrees.
    axes : tuple of 2 ints, optional
        The two axes that define the plane of rotation. Default is the first
        two axes.
    reshape : bool, optional
        If `reshape` is true, the output shape is adapted so that the input
        array is contained completely in the output. Default is True.
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 1.0.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'constant'.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0
    prefilter : bool, optional
        The parameter prefilter determines if the input is pre-filtered with
        `spline_filter` before interpolation (necessary for spline
        interpolation of order > 1).  If False, it is assumed that the input is
        already filtered. Default is True.

    Returns
    -------
    rotate : ndarray or None
        The rotated input. If `output` is given as a parameter, None is
        returned.

    """
    input = np.asarray(input)
    axes = list(axes)
    rank = input.ndim
    if axes[0] < 0:
        axes[0] += rank
    if axes[1] < 0:
        axes[1] += rank
    if axes[0] < 0 or axes[1] < 0 or axes[0] > rank or axes[1] > rank:
        raise RuntimeError('invalid rotation plane specified')
    if axes[0] > axes[1]:
        axes = axes[1], axes[0]
    angle = np.pi / 180 * angle
    m11 = math.cos(angle)
    m12 = math.sin(angle)
    m21 = -math.sin(angle)
    m22 = math.cos(angle)
    matrix = np.array([[m11, m12],
                             [m21, m22]], dtype = np.float64)
    iy = input.shape[axes[0]]
    ix = input.shape[axes[1]]
    if reshape:
        mtrx = np.array([[ m11, -m21],
                               [-m12,  m22]], dtype = np.float64)
        minc = [0, 0]
        maxc = [0, 0]
        coor = np.dot(mtrx, [0, ix])
        minc, maxc = _minmax(coor, minc, maxc)
        coor = np.dot(mtrx, [iy, 0])
        minc, maxc = _minmax(coor, minc, maxc)
        coor = np.dot(mtrx, [iy, ix])
        minc, maxc = _minmax(coor, minc, maxc)
        oy = int(maxc[0] - minc[0] + 1.0)
        ox = int(maxc[1] - minc[1] + 1.0)
    else:
        oy = input.shape[axes[0]]
        ox = input.shape[axes[1]]
    offset = np.zeros((2,), dtype = np.float64)
    offset[0] = float(oy) / 2.0 - 1.0
    offset[1] = float(ox) / 2.0 - 1.0
    offset = np.dot(matrix, offset)
    tmp = np.zeros((2,), dtype = np.float64)
    tmp[0] = float(iy) / 2.0 - 1.0
    tmp[1] = float(ix) / 2.0 - 1.0
    offset = tmp - offset
    output_shape = list(input.shape)
    output_shape[axes[0]] = oy
    output_shape[axes[1]] = ox
    output_shape = tuple(output_shape)
    output, return_value = _ni_support._get_output(output, input,
                                                   shape=output_shape)
    if input.ndim <= 2:
        affine_transform(input, matrix, offset, output_shape, output,
                         order, mode, cval, prefilter)
    else:
        coordinates = []
        size = np.product(input.shape,axis=0)
        size //= input.shape[axes[0]]
        size //= input.shape[axes[1]]
        for ii in range(input.ndim):
            if ii not in axes:
                coordinates.append(0)
            else:
                coordinates.append(slice(None, None, None))
        iter_axes = list(range(input.ndim))
        iter_axes.reverse()
        iter_axes.remove(axes[0])
        iter_axes.remove(axes[1])
        os = (output_shape[axes[0]], output_shape[axes[1]])
        for ii in range(size):
            ia = input[tuple(coordinates)]
            oa = output[tuple(coordinates)]
            affine_transform(ia, matrix, offset, os, oa, order, mode,
                             cval, prefilter)
            for jj in iter_axes:
                if coordinates[jj] < input.shape[jj] - 1:
                    coordinates[jj] += 1
                    break
                else:
                    coordinates[jj] = 0
    return return_value


def affine_transform(input, matrix, offset=0.0, output_shape=None,
                     output=None, order=3,
                     mode='constant', cval=0.0, prefilter=True):
    """
    Apply an affine transformation.

    The given matrix and offset are used to find for each point in the
    output the corresponding coordinates in the input by an affine
    transformation. The value of the input at those coordinates is
    determined by spline interpolation of the requested order. Points
    outside the boundaries of the input are filled according to the given
    mode.

    Parameters
    ----------
    input : ndarray
        The input array.
    matrix : ndarray
        The matrix must be two-dimensional or can also be given as a
        one-dimensional sequence or array. In the latter case, it is assumed
        that the matrix is diagonal. A more efficient algorithms is then
        applied that exploits the separability of the problem.
    offset : float or sequence, optional
        The offset into the array where the transform is applied. If a float,
        `offset` is the same for each axis. If a sequence, `offset` should
        contain one value for each axis.
    output_shape : tuple of ints, optional
        Shape tuple.
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'constant'.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0
    prefilter : bool, optional
        The parameter prefilter determines if the input is pre-filtered with
        `spline_filter` before interpolation (necessary for spline
        interpolation of order > 1).  If False, it is assumed that the input is
        already filtered. Default is True.

    Returns
    -------
    affine_transform : ndarray or None
        The transformed input. If `output` is given as a parameter, None is
        returned.

    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if output_shape is None:
        output_shape = input.shape
    if input.ndim < 1 or len(output_shape) < 1:
        raise RuntimeError('input and output rank must be > 0')
    mode = _extend_mode_to_code(mode)
    if prefilter and order > 1:
        filtered = spline_filter(input, order, output = np.float64)
    else:
        filtered = input
    output, return_value = _ni_support._get_output(output, input,
                                                   shape=output_shape)
    matrix = np.asarray(matrix, dtype = np.float64)
    if matrix.ndim not in [1, 2] or matrix.shape[0] < 1:
        raise RuntimeError('no proper affine matrix provided')
    if matrix.shape[0] != input.ndim:
        raise RuntimeError('affine matrix has wrong number of rows')
    if matrix.ndim == 2 and matrix.shape[1] != output.ndim:
        raise RuntimeError('affine matrix has wrong number of columns')
    if not matrix.flags.contiguous:
        matrix = matrix.copy()
    offset = _ni_support._normalize_sequence(offset, input.ndim)
    offset = np.asarray(offset, dtype = np.float64)
    if offset.ndim != 1 or offset.shape[0] < 1:
        raise RuntimeError('no proper offset provided')
    if not offset.flags.contiguous:
        offset = offset.copy()
    if matrix.ndim == 1:
        _nd_image.zoom_shift(filtered, matrix, offset, output, order,
                             mode, cval)
    else:
        _nd_image.geometric_transform(filtered, None, None, matrix, offset,
                            output, order, mode, cval, None, None)
    return return_value


def _extend_mode_to_code(mode):
    mode = _ni_support._extend_mode_to_code(mode)
    return mode

def spline_filter(input, order=3, output = np.float64):
    """
    Multi-dimensional spline filter.

    For more details, see `spline_filter1d`.

    See Also
    --------
    spline_filter1d

    Notes
    -----
    The multi-dimensional filter is implemented as a sequence of
    one-dimensional spline filters. The intermediate arrays are stored
    in the same data type as the output. Therefore, for output types
    with a limited precision, the results may be imprecise because
    intermediate results may be stored with insufficient precision.

    """
    if order < 2 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    output, return_value = _ni_support._get_output(output, input)
    if order not in [0, 1] and input.ndim > 0:
        for axis in range(input.ndim):
            spline_filter1d(input, order, axis, output = output)
            input = output
    else:
        output[...] = input[...]
    return return_value


def spline_filter1d(input, order=3, axis=-1, output=np.float64):
    """
    Calculates a one-dimensional spline filter along the given axis.

    The lines of the array along the given axis are filtered by a
    spline filter. The order of the spline must be >= 2 and <= 5.

    Parameters
    ----------
    input : array_like
        The input array.
    order : int, optional
        The order of the spline, default is 3.
    axis : int, optional
        The axis along which the spline filter is applied. Default is the last
        axis.
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array. Default is `np.float64`.

    Returns
    -------
    spline_filter1d : ndarray or None
        The filtered input. If `output` is given as a parameter, None is
        returned.

    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    output, return_value = _ni_support._get_output(output, input)
    if order in [0, 1]:
        output[...] = np.array(input)
    else:
        axis = _ni_support._check_axis(axis, input.ndim)
        _nd_image.spline_filter1d(input, order, axis, output)
    return return_value
