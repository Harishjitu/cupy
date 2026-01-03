from __future__ import annotations

import cupy

_POLICIES = ['propagate', 'raise', 'omit']


def _validate_nan_policy(nan_policy):
    """Validate nan_policy parameter."""
    if nan_policy not in _POLICIES:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in _POLICIES))


def _check_nan_raise(a):
    """Raise error if array contains NaN."""
    if cupy.isnan(cupy.sum(a)):
        raise ValueError("The input contains nan values")


def _get_dtype(a):
    return a.dtype if a.dtype.kind in 'fc' else cupy.float64


def _pow_int(x, n):
    """Optimized integer power using exponentiation by squares.

    Parameters
    ----------
    x : cupy.ndarray
        Input array.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    cupy.ndarray
        x raised to the power n.
    """
    if n == 0:
        return cupy.ones_like(x)
    elif n == 1:
        return x.copy()

    n_list = [n]
    current_n = n
    while current_n > 2:
        if current_n % 2:
            current_n = (current_n - 1) / 2
        else:
            current_n /= 2
        n_list.append(current_n)

    if n_list[-1] == 1:
        s = x
    else:
        s = cupy.square(x)

    for exp in n_list[-2::-1]:
        s = cupy.square(s)
        if exp % 2:
            s = s * x

    return s


def _first(arr, axis):
    """Return arr[..., 0:1, ...] where 0:1 is in the `axis` position

    """

    return cupy.take_along_axis(arr, cupy.array(0, ndmin=arr.ndim), axis)


def _isconst(x):
    """Check if all values in x are the same.  nans are ignored.
    x must be a 1d array. The return value is a 1d array
    with length 1, so it can be used in cupy.apply_along_axis.

    """

    y = x[~cupy.isnan(x)]
    if y.size == 0:
        return cupy.array([True])
    else:
        return (y[0] == y).all(keepdims=True)


def zscore(a, axis=0, ddof=0, nan_policy='propagate'):
    """Compute the z-score.

    Compute the z-score of each value in the sample, relative to
    the sample mean and standard deviation.

    Parameters
    ----------
    a : array-like
        An array like object containing the sample data
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None,
        compute over the whole arrsy `a`
    ddof : int, optional
        Degrees of freedom correction in the calculation of the
        standard deviation. Default is 0
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate'
        returns nan, 'raise' throws an error, 'omit' performs
        the calculations ignoring nan values. Default is
        'propagate'. Note that when the value is 'omit',
        nans in the input also propagate to the output,
        but they do not affect the z-scores computed
        for the non-nan values

    Returns
    -------
    zscore : array-like
        The z-scores, standardized by mean and standard deviation of
        input array `a`

    """

    return zmap(a, a, axis=axis, ddof=ddof, nan_policy=nan_policy)


def zmap(scores, compare, axis=0, ddof=0, nan_policy='propagate'):
    """Calculate the relative z-scores.

    Return an array of z-scores, i.e., scores that are standardized
    to zero mean and unit variance, where mean and variance are
    calculated from the comparison array.

    Parameters
    ----------
    scores : array-like
        The input for which z-scores are calculated
    compare : array-like
        The input from which the mean and standard deviation of
        the normalization are taken; assumed to have the same
        dimension as `scores`
    axis : int or None, optional
        Axis over which mean and variance of `compare` are calculated.
        Default is 0. If None, compute over the whole array `scores`
    ddof : int, optional
        Degrees of freedom correction in the calculation of the
        standard deviation. Default is 0
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle the occurrence of nans in `compare`.
        'propagate' returns nan, 'raise' raises an exception, 'omit'
        performs the calculations ignoring nan values. Default is
        'propagate'. Note that when the value is 'omit', nans in `scores`
        also propagate to the output, but they do not affect the z-scores
        computed for the non-nan values

    Returns
    -------
    zscore : array-like
        Z-scores, in the same shape as `scores`

    """

    _validate_nan_policy(nan_policy)

    a = compare

    if a.size == 0:
        dtype = _get_dtype(a)
        return cupy.empty(a.shape, dtype)

    if nan_policy == 'raise':
        _check_nan_raise(a)

    if nan_policy == 'omit':
        if axis is None:
            mn = cupy.nanmean(a.ravel())
            std = cupy.nanstd(a.ravel(), ddof=ddof)
            isconst = _isconst(a.ravel())
        else:
            mn = cupy.nanmean(a, axis=axis, keepdims=True)
            std = cupy.nanstd(a, axis=axis, keepdims=True, ddof=ddof)
            isconst = (_first(a, axis) == a).all(axis=axis, keepdims=True)
    else:
        mn = a.mean(axis=axis, keepdims=True)
        std = a.std(axis=axis, ddof=ddof, keepdims=True)
        if axis is None:
            isconst = (a.ravel()[0] == a).all()
        else:
            isconst = (_first(a, axis) == a).all(axis=axis, keepdims=True)

    # Set std deviations that are 0 to 1 to avoid division by 0.
    std[isconst] = 1.0
    z = (scores - mn) / std

    # Set the outputs associated with a constant input to nan.
    z[cupy.broadcast_to(isconst, z.shape)] = cupy.nan
    return z


def moment(a, order=1, axis=0, nan_policy='propagate'):
    """Calculate the nth moment about the mean for a sample.

    A moment is a specific quantitative measure of the shape of a set of
    points. It is often used to calculate coefficients of skewness and
    kurtosis due to its close relationship with them.

    Parameters
    ----------
    a : array-like
        Input array
    order : int, optional
        Order of central moment that is returned. Default is 1.
    axis : int or None, optional
        Axis along which the central moment is computed. Default is 0.
        If None, compute over the whole array.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        - 'propagate': returns nan
        - 'raise': raises an error
        - 'omit': performs the calculations ignoring nan values
        Default is 'propagate'.

    Returns
    -------
    n-th central moment : ndarray or float
        The appropriate moment along the given axis or over all values
        if axis is None.
    """

    _validate_nan_policy(nan_policy)
    a = cupy.asarray(a)
    dtype = _get_dtype(a)

    if a.size == 0:
        return cupy.mean(a, axis=axis, dtype=dtype)

    if nan_policy == 'raise':
        _check_nan_raise(a)

    # 0th moment is always 1
    if order == 0:
        if axis is None:
            return dtype.type(1.0)
        shape = list(a.shape)
        del shape[axis]
        return cupy.ones(shape, dtype=dtype)

    # 1st central moment is always 0
    if order == 1:
        if axis is None:
            return dtype.type(0.0)
        shape = list(a.shape)
        del shape[axis]
        return cupy.zeros(shape, dtype=dtype)

    mean_func = cupy.nanmean if nan_policy == 'omit' else cupy.mean
    mean = mean_func(a, axis=axis, keepdims=True)
    a_zero_mean = a - mean
    powered = _pow_int(a_zero_mean, order)

    return mean_func(powered, axis=axis)


def skew(a, axis=0, bias=True, nan_policy='propagate'):
    """Compute the sample skewness of a data set.

    For normally distributed data, the skewness should be about zero.
    For unimodal continuous distributions, a skewness value greater
    than zero means that there is more weight in the right tail of
    the distribution.

    Parameters
    ----------
    a : array-like
        Input array.
    axis : int or None, optional
        Axis along which skewness is calculated. Default is 0.
        If None, compute over the whole array.
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.
        Default is True.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.

        - 'propagate': returns nan
        - 'raise': raises an error
        - 'omit': performs the calculations ignoring nan values

        Default is 'propagate'.

    Returns
    -------
    skewness : ndarray or float
        The skewness of values along an axis, returning NaN where all
        values are equal.
    """
    _validate_nan_policy(nan_policy)
    a = cupy.asarray(a)
    dtype = _get_dtype(a)

    if a.size == 0:
        return cupy.mean(a, axis=axis, dtype=dtype)

    if nan_policy == 'raise':
        _check_nan_raise(a)

    # Compute 2nd and 3rd moments
    m2 = moment(a, 2, axis, nan_policy=nan_policy)
    m3 = moment(a, 3, axis, nan_policy=nan_policy)

    # Compute skewness: m3 / m2^(3/2)
    skewness = m3 / (m2 ** 1.5)

    if not bias:
        n = a.size if axis is None else a.shape[axis]
        if n > 2:
            skewness = skewness * cupy.sqrt(n * (n - 1)) / (n - 2)

    return skewness


def kurtosis(a, axis=0, fisher=True, bias=True, nan_policy='propagate'):
    """Compute the kurtosis (Fisher or Pearson) of a dataset.

    Kurtosis is the fourth central moment divided by the square of the
    variance. If Fisher's definition is used, then 3.0 is subtracted
    from the result to give 0.0 for a normal distribution.

    If bias is False then the kurtosis is calculated using k statistics
    to eliminate bias coming from biased moment estimators.

    Parameters
    ----------
    a : array-like
        Input array.
    axis : int or None, optional
        Axis along which the kurtosis is calculated. Default is 0.
        If None, compute over the whole array.
    fisher : bool, optional
        If True, Fisher's definition is used (normal ==> 0.0).
        If False, Pearson's definition is used (normal ==> 3.0).
        Default is True.
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.
        Default is True.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.

        - 'propagate': returns nan
        - 'raise': raises an error
        - 'omit': performs the calculations ignoring nan values

        Default is 'propagate'.

    Returns
    -------
    kurtosis : ndarray or float
        The kurtosis of values along an axis, returning NaN where all
        values are equal.
    """
    _validate_nan_policy(nan_policy)
    a = cupy.asarray(a)
    dtype = _get_dtype(a)

    if a.size == 0:
        return cupy.mean(a, axis=axis, dtype=dtype)

    if nan_policy == 'raise':
        _check_nan_raise(a)

    # Compute 2nd and 4th moments
    m2 = moment(a, 2, axis, nan_policy=nan_policy)
    m4 = moment(a, 4, axis, nan_policy=nan_policy)

    # Compute kurtosis: m4 / m2^2
    kurt = m4 / (m2 ** 2)

    if not bias:
        n = a.size if axis is None else a.shape[axis]
        if n > 3:
            kurt = ((n + 1) * kurt - 3 * (n - 1)) * \
                (n - 1) / ((n - 2) * (n - 3)) + 3

    if fisher:
        kurt = kurt - 3

    return kurt
