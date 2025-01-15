import numpy as np
import numba
import xarray as xr


@numba.jit(nopython=True)
def lowess_1d(y, x, alpha=2. / 3., it=10):
    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(y)
    r = int(np.ceil(alpha * n))   
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(it):
        for i in range(n):
            h = np.sort(np.abs(x - x[i]))[r]
            dist = np.abs((x - x[i]) / h)
            dist[dist < 0.] = 0.
            dist[dist > 1.] = 1.
            w = (1 - dist ** 3) ** 3
            weights = delta * w
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = np.linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]    
        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = residuals / (6.0 * s)
        dist[dist < -1.] = -1.
        dist[dist > 1.] = 1.
        delta = (1 - delta ** 2) ** 2
    return yest

  
def lowess(obj, dim, alpha=0.5, it=5):
    """
    Apply a LOWESS smoothing along one dimension
    
    Parameters
    ----------
    obj : xarray.DataArray, xarray.Dataset 
        The input dataarray or dataset
    dim : str
        The dimension along which the computation will be performed
    alpha : float, optional
        Span of the smoothing
    it : int, optional
        Number of iterations
    
    Returns
    -------
    res : xarray.DataArray, xarray.Dataset
        The estimated lowess smoothing
    """
    if isinstance(obj, xr.DataArray):
        res = np.apply_along_axis(lowess_1d, obj.get_axis_num(dim), obj.data, 
                                  obj[dim].astype('f4').data, alpha=alpha, it=it)
        return xr.DataArray(res, coords=obj.coords, dims=obj.dims)
    elif isinstance(obj, xr.Dataset):
        return obj.map(lowess_1d, keep_attrs=True, alpha=alpha, it=it)
    else:
        raise ValueError