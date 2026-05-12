from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.tsatools import detrend
from scipy import stats
import xarray as xr
import numpy as np
import pymannkendall as mk



def loess1d(x, period):
    x = x.copy()
    res = STL(x, period=period).fit()
    return res.trend


def loess3d(x, dim, period):
    return xr.apply_ufunc(loess1d, x, input_core_dims=[[dim]], output_core_dims=[[dim]], kwargs=dict(period=period), vectorize=True, dask="parallelized")


# detrending and anomaly calculation

def detrend1d_check(arr, period):
    # if np.nansum(arr) != 0:
    #     trend = mk.hamed_rao_modification_test(arr).trend
    # else:
    #     trend = 'no trend'
    # if trend != 'no trend':
    #     res = STL(arr, period = period).fit()
    #     arr_det = arr - res.trend
    #     return arr_det
    # else:
    #     return arr
    arr = arr.copy() # to counter buffer is read-only error
    res = STL(arr, period = period).fit()
    arr_det = arr - res.trend
    # arr_det = detrend(arr, order=2)
    return arr_det

def detrend_separate_check(da, dim, period):
    # da = da.copy()
    return xr.apply_ufunc(detrend1d_check, da, input_core_dims=[[dim]], output_core_dims=[[dim]], kwargs=dict(period=period), vectorize=True, dask='parallelized')



def detrend_rolling_window(da, window_size=30):
    pad_size=window_size//2
    padded_data = da.pad(time=(pad_size, pad_size), mode='edge')
    smoothed_data = padded_data.rolling(time=window_size, center=True).mean('time').isel(time = slice(int(window_size/2),-int(window_size/2)))
    return da - smoothed_data



def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da.coords[dim], p.polyfit_coefficients)
    return da - fit


def calc_anom(
    input_da,
    base_clim
):

    da_clim = base_clim.groupby("time.month").mean("time")
    da_anom = input_da.groupby("time.month") - da_clim
    
    return da_anom#, da_clim


def calc_anom_annual(
    input_da,
    base_clim
):

    da_clim = base_clim.mean("time")
    da_anom = input_da - da_clim
    
    return da_anom#, da_clim


# functions to calculate correlation value
def get_corr_1d(x, y):
    xx = x[~(np.isnan(x) & np.isnan(y))]
    yy = y[~(np.isnan(x) & np.isnan(y))]
    statistic, _ = stats.spearmanr(xx, yy)
    return statistic

def get_regr_1d(x, y):
    xx = x[~(np.isnan(x) & np.isnan(y))]
    yy = y[~(np.isnan(x) & np.isnan(y))]
    res = stats.linregress(xx, yy)
    return res.slope


def get_pval_1d(x, y):
    xx = x[~(np.isnan(x) & np.isnan(y))]
    yy = y[~(np.isnan(x) & np.isnan(y))]
    _, pval = stats.spearmanr(xx, yy)
    return pval


def get_corr(x, y, dim):
    return xr.apply_ufunc(get_corr_1d, x, y, input_core_dims=[[dim], [dim]], vectorize=True, dask = 'parallelized')

def get_regr(x, y, dim):
    return xr.apply_ufunc(get_regr_1d, x, y, input_core_dims=[[dim], [dim]], vectorize=True, dask = 'parallelized')

def get_pval(x, y, dim):
    return xr.apply_ufunc(get_pval_1d, x, y, input_core_dims=[[dim], [dim]], vectorize=True, dask = 'parallelized')



### combiene the individual operations above into one neat function
def telemetrics_1d(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    xx = x[mask]
    yy = y[mask]

    if len(xx) < 3:
        return np.nan, np.nan, np.nan

    # regression
    slope, _, _, _, _ = stats.linregress(xx, yy)

    # spearman corr + pval
    corr, pval = stats.spearmanr(xx, yy)

    return slope, corr, pval



def telemetrics_fast(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    xx = x[mask]
    yy = y[mask]

    n = len(xx)
    if n < 3:
        return np.nan, np.nan, np.nan

    # ----------- regression slope (NumPy) -----------
    x_mean = xx.mean()
    y_mean = yy.mean()
    slope = np.dot(xx - x_mean, yy - y_mean) / np.dot(xx - x_mean, xx - x_mean)

    # ----------- Spearman corr (NumPy) -----------
    # Rank (argsort twice) – MUCH faster than scipy.spearmanr
    rx = np.argsort(np.argsort(xx))
    ry = np.argsort(np.argsort(yy))

    rx = rx - rx.mean()
    ry = ry - ry.mean()
    corr = np.dot(rx, ry) / np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))

    # p-value is very expensive → use analytic t-approximation
    df = n - 2
    t = corr * np.sqrt(df / (1 - corr**2))
    pval = 2 * stats.t.sf(np.abs(t), df=df)

    return slope, corr, pval



# def get_all_metrics(x, y, dim):
#     return xr.apply_ufunc(
#         telemetrics_1d,
#         x, y,
#         input_core_dims=[[dim], [dim]],
#         output_core_dims=[[], [], []],
#         vectorize=True,
#         dask="parallelized",
#         output_dtypes=[np.float64, np.float64, np.float64],
#     )


def get_all_metrics(x, y, dim):
    return xr.apply_ufunc(
        telemetrics_fast,
        x, y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64],
    )



def mannwhitneyu1d(dist1, dist2):
    # guard against empty or constant samples
    if len(dist1) < 2 or len(dist2) < 2:
        return np.nan
    return float(stats.mannwhitneyu(dist1, dist2).pvalue)

def mannwhitneyu3d(da1, da2, dim):
    return xr.apply_ufunc(
        mannwhitneyu1d, da1, da2,
        input_core_dims=[[dim], [dim]],
        exclude_dims={dim},
        vectorize=True,
        dask='allowed'
    )


def calc_trend1d(x: np.ndarray):
    if np.nansum(x) != 0:
        res = mk.hamed_rao_modification_test(x)
        return res.slope*10
    else:
        return np.nan


def calc_trend_pval1d(x: np.ndarray):
    if np.nansum(x) != 0:
        res = mk.hamed_rao_modification_test(x)
        return res.p
    else:
        return np.nan


def calc_trend3d(da: xr.DataArray, dim: str):
    if not isinstance(da, xr.DataArray):
        raise TypeError("'xr.DataArray' input is required, not the %s" % (type(da)))
    return xr.apply_ufunc(calc_trend1d, da, input_core_dims=[[dim]], vectorize=True, dask='parallelized')


def calc_trend_pval3d(da: xr.DataArray, dim: str):
    if not isinstance(da, xr.DataArray):
        raise TypeError("'xr.DataArray' input is required, not the %s" % (type(da)))
    return xr.apply_ufunc(calc_trend_pval1d, da, input_core_dims=[[dim]], vectorize=True, dask='parallelized')