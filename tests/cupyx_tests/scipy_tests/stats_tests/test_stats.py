from __future__ import annotations

import numpy
import pytest

import cupy
from cupy import testing
import cupyx
import cupyx.scipy.stats  # NOQA

try:
    import scipy.stats
except ImportError:
    pass


atol = {'default': 1e-6, cupy.float64: 1e-14}
rtol = {'default': 1e-6, cupy.float64: 1e-14}


@testing.with_requires('scipy')
class TestTrim:

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        scipy_name='scp', rtol=1e-6, contiguous_check=False)
    @pytest.mark.parametrize('shape', [(24,), (6, 4), (6, 4, 3), (4, 6)])
    def test_base(self, xp, scp, dtype, order, shape):
        a = testing.shaped_random(
            shape, xp=xp, dtype=dtype, order=order, scale=100)
        return scp.stats.trim_mean(a, 2 / 6.)

    @testing.for_all_dtypes()
    def test_zero_dim(self, dtype):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            a = xp.array(0, dtype=dtype)
            with pytest.raises(IndexError):
                return scp.stats.trim_mean(a, 2 / 6.)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_dim_axis_none(self, xp, scp, dtype):
        a = xp.array(0, dtype=dtype)
        return scp.stats.trim_mean(a, 2 / 6., axis=None)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(scipy_name='scp')
    @pytest.mark.parametrize('propotiontocut', [0.0, 0.6])
    def test_empty(self, xp, scp, dtype, propotiontocut):
        a = xp.array([])
        return scp.stats.trim_mean(a, 2 / 6., propotiontocut)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        scipy_name='scp', rtol=1e-6, contiguous_check=False)
    @pytest.mark.parametrize('axis', [0, 1, 2, 3, -1, None])
    def test_axis(self, xp, scp, dtype, order, axis):
        a = testing.shaped_random(
            (5, 6, 4, 7), xp=xp, dtype=dtype, order=order, scale=100)
        return scp.stats.trim_mean(a, 2 / 6., axis=axis)

    def test_propotion_too_big(self):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            a = xp.array([4, 8, 2, 0, 9, 5, 10, 1, 7, 3, 6])
            with pytest.raises(ValueError):
                scp.stats.trim_mean(a, 0.6)


@testing.with_requires('scipy')
class TestZmap:

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_1dim(self, xp, scp, dtype):
        x = testing.shaped_random((10,), xp, dtype=dtype)
        y = testing.shaped_random((8,), xp, dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_2dim(self, xp, scp, dtype):
        x = testing.shaped_random((2, 6), xp, dtype=dtype)
        y = testing.shaped_random((2, 1), xp, dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_multi_dim(self, xp, scp, dtype):
        x = testing.shaped_random((3, 4, 5, 7), xp, dtype=dtype)
        y = testing.shaped_random((3, 4, 1, 1), xp, dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_multi_dim_2(self, xp, scp, dtype):
        x = testing.shaped_random((4, 4, 5, 6, 2), xp, dtype=dtype)
        y = testing.shaped_random((4, 4, 5, 6, 2), xp, dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-3, rtol=1e-3)
    def test_zmap_multi_dim_2_float16(self, xp, scp):
        x = testing.shaped_random((4, 4, 5, 6, 2), xp, dtype=xp.float16)
        y = testing.shaped_random((4, 4, 5, 6, 2), xp, dtype=xp.float16)
        return scp.stats.zmap(x, y)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_with_axis(self, xp, scp, dtype):
        x = testing.shaped_random((2, 3), xp, dtype=dtype)
        y = testing.shaped_random((1, 3), xp, dtype=dtype)
        return scp.stats.zmap(x, y, axis=1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zmap_with_axis_ddof(self, xp, scp, dtype):
        x = testing.shaped_random((4, 5), xp, dtype=dtype)
        y = testing.shaped_random((1, 5), xp, dtype=dtype)
        return scp.stats.zmap(x, y, axis=1, ddof=2)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zmap_empty(self, xp, scp, dtype):
        x = xp.array([], dtype=dtype)
        y = xp.array([1, 3, 5], dtype=dtype)
        return scp.stats.zmap(x, y)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zmap_nan_policy_propagate(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        y = xp.array([xp.nan, -4.0, -1.0, -5.0], dtype=dtype)
        with numpy.errstate(invalid='ignore'):  # numpy warns with complex
            return scp.stats.zmap(x, y, nan_policy='propagate')

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zmap_nan_policy_omit(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        y = xp.array([xp.nan, -4.0, -1.0, -5.0], dtype=dtype)
        return scp.stats.zmap(x, y, nan_policy='omit')

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zmap_nan_policy_omit_axis_ddof(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        y = xp.array([xp.nan, -4.0, -1.0, -5.0], dtype=dtype)
        return scp.stats.zmap(x, y, axis=0, ddof=1, nan_policy='omit')

    @testing.for_dtypes('fdFD')
    @testing.with_requires('scipy>=1.7')
    def test_zmap_nan_policy_raise(self, dtype):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            x = xp.array([1, 2, 3], dtype=dtype)
            y = xp.array([8, -4, xp.nan, 4], dtype=dtype)
            with pytest.raises(ValueError):
                scp.stats.zmap(x, y, nan_policy='raise')


@testing.with_requires('scipy')
class TestZscore:

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zscore_1dim(self, xp, scp, dtype):
        x = testing.shaped_random((10,), xp, dtype=dtype)
        return scp.stats.zscore(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-3)
    def test_zscore_1dim_float16(self, xp, scp):
        x = testing.shaped_random((10,), xp, dtype=xp.float16)
        return scp.stats.zscore(x)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zscore_2dim(self, xp, scp, dtype):
        x = testing.shaped_random((5, 3), xp, dtype=dtype)
        return scp.stats.zscore(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-3, rtol=1e-3)
    def test_zscore_2dim_float16(self, xp, scp):
        x = testing.shaped_random((5, 3), xp, dtype=xp.float16)
        return scp.stats.zscore(x)

    @testing.for_all_dtypes(no_float16=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zscore_multi_dim(self, xp, scp, dtype):
        x = testing.shaped_random((3, 4, 5, 7), xp, dtype=dtype)
        return scp.stats.zscore(x)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-3, rtol=1e-3)
    def test_zscore_multi_dim_float16(self, xp, scp):
        x = testing.shaped_random((3, 4, 5, 7), xp, dtype=xp.float16)
        return scp.stats.zscore(x)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=atol)
    def test_zscore_with_axis(self, xp, scp, dtype):
        x = testing.shaped_random((5, 6), xp, dtype=dtype)
        return scp.stats.zscore(x, axis=1)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_zscore_with_axis_ddof(self, xp, scp, dtype):
        x = testing.shaped_random((2, 3, 8), xp, dtype=dtype)
        return scp.stats.zscore(x, axis=2, ddof=2)

    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-3, rtol=1e-3)
    def test_zscore_with_axis_ddof_float16(self, xp, scp):
        x = testing.shaped_random((2, 3, 8), xp, dtype=xp.float16)
        return scp.stats.zscore(x, axis=2, ddof=2)

    @testing.with_requires('scipy>=1.15')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zscore_empty(self, xp, scp, dtype):
        x = xp.array([], dtype=dtype)
        return scp.stats.zscore(x)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zscore_nan_policy_propagate(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        with numpy.errstate(invalid='ignore'):  # numpy warns with complex
            return scp.stats.zscore(x, nan_policy='propagate')

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zscore_nan_policy_omit(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        return scp.stats.zscore(x, nan_policy='omit')

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @testing.with_requires('scipy>=1.7')
    def test_zscore_nan_policy_omit_axis_ddof(self, xp, scp, dtype):
        x = xp.array([4.0, 1.0, 1.0, xp.nan], dtype=dtype)
        return scp.stats.zscore(x, axis=0, ddof=1, nan_policy='omit')

    @testing.for_dtypes('fdFD')
    @testing.with_requires('scipy>=1.7')
    def test_zscore_nan_policy_raise(self, dtype):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            x = xp.array([1, 2, 3, xp.nan], dtype=dtype)
            with pytest.raises(ValueError):
                scp.stats.zscore(x, nan_policy='raise')


@testing.with_requires('scipy')
class TestMoment:

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_moment_1dim(self, xp, scp, dtype):
        x = testing.shaped_random((100,), xp, dtype=dtype)
        return scp.stats.moment(x, order=2)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_moment_2dim(self, xp, scp, dtype):
        x = testing.shaped_random((10, 20), xp, dtype=dtype)
        return scp.stats.moment(x, order=2)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-5, rtol=1e-5)
    @pytest.mark.parametrize('order', [0, 1, 2, 3, 4, 5])
    def test_moment_orders(self, xp, scp, dtype, order):
        x = testing.shaped_random((50,), xp, dtype=dtype)
        return scp.stats.moment(x, order=order)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @pytest.mark.parametrize('axis', [0, 1, None])
    def test_moment_axis(self, xp, scp, dtype, axis):
        x = testing.shaped_random((10, 20), xp, dtype=dtype)
        return scp.stats.moment(x, order=2, axis=axis)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_moment_empty(self, xp, scp, dtype):
        x = xp.array([], dtype=dtype)
        return scp.stats.moment(x, order=2)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_moment_nan_policy_propagate(self, xp, scp, dtype):
        x = xp.array([1.0, 2.0, xp.nan, 4.0], dtype=dtype)
        return scp.stats.moment(x, order=2, nan_policy='propagate')

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_moment_nan_policy_omit(self, xp, scp, dtype):
        x = xp.array([1.0, 2.0, xp.nan, 4.0], dtype=dtype)
        return scp.stats.moment(x, order=2, nan_policy='omit')

    @testing.for_dtypes('fd')
    def test_moment_nan_policy_raise(self, dtype):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            x = xp.array([1.0, 2.0, xp.nan, 4.0], dtype=dtype)
            with pytest.raises(ValueError):
                scp.stats.moment(x, order=2, nan_policy='raise')


@testing.with_requires('scipy')
class TestSkew:

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_skew_1dim(self, xp, scp, dtype):
        x = testing.shaped_random((100,), xp, dtype=dtype)
        return scp.stats.skew(x)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_skew_2dim(self, xp, scp, dtype):
        x = testing.shaped_random((10, 20), xp, dtype=dtype)
        return scp.stats.skew(x)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @pytest.mark.parametrize('axis', [0, 1, None])
    def test_skew_axis(self, xp, scp, dtype, axis):
        x = testing.shaped_random((10, 20), xp, dtype=dtype)
        return scp.stats.skew(x, axis=axis)

    @testing.for_all_dtypes(no_bool=True, no_complex=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=1e-5, rtol=1e-5)
    @pytest.mark.parametrize('bias', [True, False])
    def test_skew_bias(self, xp, scp, dtype, bias):
        x = testing.shaped_random((50,), xp, dtype=dtype)
        return scp.stats.skew(x, bias=bias)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_skew_empty(self, xp, scp, dtype):
        x = xp.array([], dtype=dtype)
        return scp.stats.skew(x)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_skew_constant(self, xp, scp, dtype):
        x = xp.array([5.0, 5.0, 5.0, 5.0], dtype=dtype)
        return scp.stats.skew(x)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_skew_nan_policy_propagate(self, xp, scp, dtype):
        x = xp.array([1.0, 2.0, xp.nan, 4.0], dtype=dtype)
        return scp.stats.skew(x, nan_policy='propagate')

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_skew_nan_policy_omit(self, xp, scp, dtype):
        x = xp.array([1.0, 2.0, xp.nan, 4.0], dtype=dtype)
        return scp.stats.skew(x, nan_policy='omit')

    @testing.for_dtypes('fd')
    def test_skew_nan_policy_raise(self, dtype):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            x = xp.array([1.0, 2.0, xp.nan, 4.0], dtype=dtype)
            with pytest.raises(ValueError):
                scp.stats.skew(x, nan_policy='raise')


@testing.with_requires('scipy')
class TestKurtosis:

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_kurtosis_1dim(self, xp, scp, dtype):
        x = testing.shaped_random((100,), xp, dtype=dtype)
        return scp.stats.kurtosis(x)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_kurtosis_2dim(self, xp, scp, dtype):
        x = testing.shaped_random((10, 20), xp, dtype=dtype)
        return scp.stats.kurtosis(x)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @pytest.mark.parametrize('axis', [0, 1, None])
    def test_kurtosis_axis(self, xp, scp, dtype, axis):
        x = testing.shaped_random((10, 20), xp, dtype=dtype)
        return scp.stats.kurtosis(x, axis=axis)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @pytest.mark.parametrize('fisher', [True, False])
    def test_kurtosis_fisher(self, xp, scp, dtype, fisher):
        x = testing.shaped_random((50,), xp, dtype=dtype)
        return scp.stats.kurtosis(x, fisher=fisher)

    @testing.for_all_dtypes(no_bool=True, no_complex=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @pytest.mark.parametrize('bias', [True, False])
    def test_kurtosis_bias(self, xp, scp, dtype, bias):
        x = testing.shaped_random((50,), xp, dtype=dtype)
        return scp.stats.kurtosis(x, bias=bias)

    @testing.for_all_dtypes(no_bool=True, no_complex=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    @pytest.mark.parametrize('fisher', [True, False])
    @pytest.mark.parametrize('bias', [True, False])
    def test_kurtosis_fisher_bias(self, xp, scp, dtype, fisher, bias):
        x = testing.shaped_random((50,), xp, dtype=dtype)
        return scp.stats.kurtosis(x, fisher=fisher, bias=bias)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_kurtosis_empty(self, xp, scp, dtype):
        x = xp.array([], dtype=dtype)
        return scp.stats.kurtosis(x)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_kurtosis_constant(self, xp, scp, dtype):
        x = xp.array([5.0, 5.0, 5.0, 5.0], dtype=dtype)
        return scp.stats.kurtosis(x)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_kurtosis_nan_policy_propagate(self, xp, scp, dtype):
        x = xp.array([1.0, 2.0, xp.nan, 4.0], dtype=dtype)
        return scp.stats.kurtosis(x, nan_policy='propagate')

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_allclose(scipy_name='scp', atol=atol, rtol=rtol)
    def test_kurtosis_nan_policy_omit(self, xp, scp, dtype):
        x = xp.array([1.0, 2.0, xp.nan, 4.0], dtype=dtype)
        return scp.stats.kurtosis(x, nan_policy='omit')

    @testing.for_dtypes('fd')
    def test_kurtosis_nan_policy_raise(self, dtype):
        for xp, scp in [(numpy, scipy), (cupy, cupyx.scipy)]:
            x = xp.array([1.0, 2.0, xp.nan, 4.0], dtype=dtype)
            with pytest.raises(ValueError):
                scp.stats.kurtosis(x, nan_policy='raise')
