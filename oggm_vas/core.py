""" Implementation of the 'original' volume/area scaling glacier model from
Marzeion et. al. 2012, see http://www.the-cryosphere.net/6/1295/2012/.
While the mass balance model is comparable to OGGMs past mass balance model,
the 'dynamic' part does not include any ice physics but works with area/volume
and length/volume scaling instead.

Author: Moritz Oberrauch
"""
# Built ins
import os
import json
import logging
import datetime
from time import gmtime, strftime

# External libs
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
from scipy.optimize import minimize_scalar, brentq
from sklearn.linear_model import LinearRegression

# import OGGM modules
import oggm
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH

from oggm import __version__

from oggm import utils, entity_task, global_task, workflow
from oggm.utils import floatyear_to_date, ncDataset
from oggm.exceptions import InvalidParamsError, MassBalanceCalibrationError, \
    InvalidWorkflowError

from oggm.core import climate
from oggm.core.massbalance import MassBalanceModel

# Module logger
log = logging.getLogger(__name__)

# Tolerance for the Brent's optimization method
_brentq_xtol = 2e-12

# Climate parameters relevant for the mass balance calibration
MB_PARAMS = ['temp_default_gradient', 'temp_all_solid', 'temp_all_liq',
             'temp_melt', 'prcp_scaling_factor', 'prcp_default_gradient',
             'climate_qc_months', 'hydro_month_nh', 'hydro_month_sh']


def initialize(**kwargs):
    """Calls OGGM's cfg.initialize() and adds VAS specific parameters.
    Should always be called before anything else.

    Parameters
    ----------
    kwargs
        Keyword arguments are passed to OGGM's cfg.initialize()

    """

    # call the oggm initialization
    cfg.initialize(**kwargs)

    # add precipitation lapse rate of 4%/100m from Malles and Marzeion (2021)
    cfg.PARAMS['prcp_default_gradient'] = 0.04e-2

    # area-volume scaling parameters for glaciers (cp. Marzeion et. al., 2012)
    # units: m^(3-2*gamma) and without unit, respectively
    cfg.PARAMS['vas_c_area_m2'] = 0.1912
    cfg.PARAMS['vas_gamma_area'] = 1.375

    # area-length scaling parameters for glaciers (cp. Marzeion et. al., 2012)
    # units: m^(3-q) and without unit, respectively
    cfg.PARAMS['vas_c_length_m'] = 4.5214
    cfg.PARAMS['vas_q_length'] = 2.2

    # area-volume scaling parameters for ice caps (cp. Marzeion et. al., 2012)
    # units: m^(3-2*gamma) and without unit, respectively
    cfg.PARAMS['vas_c_icecap_area_m2'] = 1.7013
    cfg.PARAMS['vas_gamma_icecap_area'] = 1.25

    # area-length scaling parameters for ice caps (cp. Marzeion et. al., 2012)
    # units: m^(3-q) and without unit, respectively
    cfg.PARAMS['vas_c_icecap_length_m'] = 7.1214
    cfg.PARAMS['vas_q_icecap_length'] = 2.5


def get_ref_tstars_filepath(fname):
    """ Returns absolute path to given file within repository.

    Parameters
    ----------
    fname : str
        filename

    Returns
    -------
    str
        absolute path to given file

    """
    fp = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                      'data', fname)
    if not os.path.isfile(fp):
        raise InvalidParamsError('File {} does not exist in this '
                                 'repository'.format(fname))
    return fp


def compute_temp_terminus(temp, temp_grad, ref_hgt,
                          terminus_hgt, temp_anomaly=0):
    """Computes the (monthly) mean temperature at the glacier terminus,
    following section 2.1.2 of Marzeion et. al., 2012. The input temperature
    is scaled by the given temperature gradient and the elevation difference
    between reference altitude and the glacier terminus elevation.

    Parameters
    ----------
    temp : netCDF4 variable
        monthly mean climatological temperature (degC)
    temp_grad : netCDF4 variable or float
        temperature lapse rate [degC per m of elevation change]
    ref_hgt : float
        reference elevation for climatological temperature [m asl.]
    terminus_hgt : float
        elevation of the glacier terminus (m asl.)
    temp_anomaly : netCDF4 variable or float, optional
        monthly mean temperature anomaly, default 0

    Returns
    -------
    netCDF4 variable
        monthly mean temperature at the glacier terminus [degC]

    """
    temp_terminus = temp + temp_grad * (terminus_hgt - ref_hgt) + temp_anomaly
    return temp_terminus


def compute_solid_prcp(prcp, prcp_factor, ref_hgt, min_hgt, max_hgt,
                       temp_terminus, temp_all_solid, temp_grad,
                       prcp_grad=0, prcp_anomaly=0):
    """Compute the (monthly) amount of solid precipitation onto the glacier
    surface, following section 2.1.1 of Marzeion et. al., 2012. The fraction of
    solid precipitation depends mainly on the terminus temperature and the
    temperature thresholds for solid and liquid precipitation. It is possible
    to scale the precipitation amount from the reference elevation to the
    average glacier surface elevation given a gradient (zero per default).

    Parameters
    ----------
    prcp : netCDF4 variable
        monthly mean climatological precipitation [kg/m2]
    prcp_factor : float
        precipitation scaling factor []
    ref_hgt : float
        reference elevation for climatological precipitation [m asl.]
    min_hgt : float
        minimum glacier elevation [m asl.]
    max_hgt : float
        maximum glacier elevation [m asl.]
    temp_terminus : netCDF4 variable
        monthly mean temperature at the glacier terminus [degC]
    temp_all_solid : float
        temperature threshold below which all precipitation is solid [degC]
    temp_grad : netCDF4 variable or float
        temperature lapse rate [degC per m of elevation change]
    prcp_grad : netCDF4 variable or float, optional
        precipitation lapse rate [percentage of precipitation per meters of
            elevation change], default = 0
    prcp_anomaly : netCDF4 variable or float, optional
        monthly mean precipitation anomaly [kg/m2], default = 0

    Returns
    -------
    netCDF4 variable
        monthly mean solid precipitation [kg/m2]

    """
    # compute fraction of solid precipitation
    if max_hgt == min_hgt:
        # prevent division by zero if max_hgt equals min_hgt
        f_solid = (temp_terminus <= temp_all_solid).astype(int)
    else:
        # use scaling defined in paper
        f_solid = (1 + (temp_terminus - temp_all_solid)
                   / (temp_grad * (max_hgt - min_hgt)))
        f_solid = np.clip(f_solid, 0, 1)

    # compute mean elevation
    mean_hgt = 0.5 * (min_hgt + max_hgt)
    # apply precipitation scaling factor
    prcp_solid = (prcp_factor * prcp + prcp_anomaly)
    # compute solid precipitation
    prcp_solid *= (1 + prcp_grad * (mean_hgt - ref_hgt)) * f_solid

    return prcp_solid


def get_min_max_elevation(gdir):
    """Reads the DEM and computes the minimal and maximal glacier surface
     elevation in meters asl, from the given (RGI) glacier outline.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`

    Returns
    -------
    [float, float]
        minimal and maximal glacier surface elevation [m asl.]

    """
    # open DEM file and mask the glacier surface area
    fpath = gdir.get_filepath('gridded_data')
    with ncDataset(fpath) as nc:
        mask = nc.variables['glacier_mask'][:]
        topo = nc.variables['topo'][:]
    # get relevant elevation information
    min_elev = np.min(topo[np.where(mask == 1)])
    max_elev = np.max(topo[np.where(mask == 1)])

    return min_elev, max_elev


def get_scaling_constant(gdirs):
    """ The scaling constants (c_l and c_a for volume/length and volume/area
    scaling respectively) are random variables and vary from glacier to
    glacier. This function computes these constants for the given glaciers,
    based on the RGI area, the inversion volume and the flowline length. Works
    for glaciers and ice caps equally. Returns dictionary with parameters.

    NOTE: The scaling constants are theoretically not independent of each
    other. Here, the scaling constants are estimated separately via a linear
    regression. So don't expect to get the same volume if you apply V/A scaling
    and V/L scaling to the same glacier.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects

    Returns
    -------
    {float, float}
        volume/length and volume/area scaling constants

    """
    # get glacier geometries
    glacier_stats = workflow.execute_entity_task(utils.glacier_statistics,
                                                 gdirs)
    rgi_id = [gs.get('rgi_id', np.NaN) for gs in glacier_stats]
    length = [gs.get('longuest_centerline_km', np.NaN) * 1e3
              for gs in glacier_stats]
    area = [gs.get('rgi_area_km2', np.NaN) * 1e6 for gs in glacier_stats]
    volume = [gs.get('inv_volume_km3', np.NaN) * 1e9 for gs in glacier_stats]
    # create DataFrame
    df = pd.DataFrame({'length': length, 'area': area, 'volume': volume},
                      index=pd.Index(rgi_id, name='rgi_id'))
    # drop glaciers where one of the geometries is missing
    df = df.dropna()
    # linear regression in log-log space for given slope
    c_l = np.exp(np.mean(np.log(df.volume.values)
                         - (cfg.PARAMS['vas_q_length']
                            * np.log(df.length.values))))
    c_a = np.exp(np.mean(np.log(df.volume.values)
                         - (cfg.PARAMS['vas_gamma_area']
                            * np.log(df.area.values))))

    return {'c_l': c_l, 'c_a': c_a}


def get_scaling_constant_exponent(gdirs, glacier_type='Glacier'):
    """ Compute scaling constants and exponent from a linear regression in
    log-log space. Returns scaling constant, scaling exponent and r squared
    for the volume/length scaling and volume/area scaling in dictionary.
    This can be done for all glaciers or all ice caps

    NOTE: The scaling parameters are theoretically not independent of each
    other. Here, they are separately estimated via a linear regression. So
    don't expect to get the same volume if you apply V/A scaling and V/L
    scaling to the same glacier.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
    glacier_type: str, optional, default='Glacier
        select between glaciers and ice caps

    Returns
    -------
    [(float, float, float), (float, float, float)]
        scaling constant, scaling exponent and r squared for the volume/length
        scaling and volume/area scaling, respectively.


    """
    # get glacier geometries
    glacier_stats = workflow.execute_entity_task(utils.glacier_statistics,
                                                 gdirs)
    rgi_id = [gs.get('rgi_id', np.NaN) for gs in glacier_stats]
    length = [gs.get('longuest_centerline_km', np.NaN) * 1e3
              for gs in glacier_stats]
    area = [gs.get('rgi_area_km2', np.NaN) * 1e6 for gs in glacier_stats]
    volume = [gs.get('inv_volume_km3', np.NaN) * 1e9 for gs in glacier_stats]
    glacier_type_list = [gs.get('glacier_type', np.NaN) for gs in
                         glacier_stats]
    # create DataFrame
    df = pd.DataFrame({'length': length, 'area': area, 'volume': volume,
                       'glacier_type': glacier_type_list},
                      index=pd.Index(rgi_id, name='rgi_id'))
    # drop glaciers where one of the geometries is missing
    df = df.dropna()
    # select for glaciers or ice caps
    df = df[df.glacier_type == glacier_type]

    # volume/length linear regression in log-log space
    x = np.log(df.length.values).reshape(-1, 1)
    y = np.log(df.volume.values)
    lin_mod = LinearRegression()
    lin_mod.fit(x, y)
    c_l = np.exp(lin_mod.intercept_)
    q = lin_mod.coef_[0]
    r_sq_l = lin_mod.score(x, y)

    # volume/area linear regression in log-log space
    x = np.log(df.area.values).reshape(-1, 1)
    y = np.log(df.volume.values)
    lin_mod = LinearRegression()
    lin_mod.fit(x, y)
    c_a = np.exp(lin_mod.intercept_)
    gamma = lin_mod.coef_[0]
    r_sq_a = lin_mod.score(x, y)

    return {'c_l': c_l, 'c_a': c_a, 'q': q, 'gamma': gamma,
            'r_sq_l': r_sq_l, 'r_sq_a': r_sq_a}


def get_yearly_mb_temp_prcp(gdir, time_range=None, year_range=None):
    """Read climate file and compute mass balance relevant climate parameters.
    Those are the positive melting temperature at glacier terminus elevation
    as energy input and the amount of solid precipitation onto the glacier
    surface as mass input. Both parameters are computes as yearly sums.

    Default is to read all data, but it is possible to specify a time range by
    giving two (included) datetime bounds. Similarly, the year range limits the
    returned data to the given bounds of (hydrological) years.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    time_range : datetime tuple, optional
        [t0, t1] time bounds, default = None
    year_range : float tuple, optional
        [y0, y1] year range, default = None

    Returns
    -------
    [float array, float array, float array]
        hydrological years (index), melting temperature [degC],
        solid precipitation [kg/m2]

    """
    # convert hydrological year range into time range
    if year_range is not None:
        sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
        em = sm - 1 if (sm > 1) else 12
        t0 = datetime.datetime(year_range[0] - 1, sm, 1)
        t1 = datetime.datetime(year_range[1], em, 1)
        return get_yearly_mb_temp_prcp(gdir, time_range=[t0, t1])

    # get needed parameters
    temp_all_solid = cfg.PARAMS['temp_all_solid']
    temp_melt = cfg.PARAMS['temp_melt']
    prcp_fac = cfg.PARAMS['prcp_scaling_factor']
    default_grad = cfg.PARAMS['temp_default_gradient']
    g_minmax = cfg.PARAMS['temp_local_gradient_bounds']
    # Marzeion et. al. (2012) used a precipitation lapse rate of 3%/100m.
    # Malles and Marzeion (2021) used a precipitation lapse rate of 4%/100m.
    prcp_grad = cfg.PARAMS['prcp_default_gradient']

    # read the climate file
    igrad = None
    with utils.ncDataset(gdir.get_filepath('climate_historical')) as nc:
        # time
        time = nc.variables['time']
        time = netCDF4.num2date(time[:], time.units)
        # limit data to given time range and
        # raise errors is bounds are outside available data
        if time_range is not None:
            p0 = np.where(time == time_range[0])[0]
            try:
                p0 = p0[0]
            except IndexError:
                raise climate.MassBalanceCalibrationError('time_range[0] '
                                                          'not found in file')
            p1 = np.where(time == time_range[1])[0]
            try:
                p1 = p1[0]
            except IndexError:
                raise climate.MassBalanceCalibrationError('time_range[1] not '
                                                          'found in file')
        else:
            p0 = 0
            p1 = len(time) - 1

        time = time[p0:p1 + 1]

        # read time series of temperature and precipitation
        itemp = nc.variables['temp'][p0:p1 + 1]
        iprcp = nc.variables['prcp'][p0:p1 + 1]
        # read time series of temperature lapse rate
        if 'gradient' in nc.variables:
            igrad = nc.variables['gradient'][p0:p1 + 1]
            # Security for stuff that can happen with local gradients
            igrad = np.where(~np.isfinite(igrad), default_grad, igrad)
            igrad = np.clip(igrad, g_minmax[0], g_minmax[1])
        # read climate data reference elevation
        ref_hgt = nc.ref_hgt

    # use the default gradient if no gradient is supplied by the climate file
    if igrad is None:
        igrad = itemp * 0 + default_grad

    # Up to this point, the code is mainly copy and paste from the
    # corresponding OGGM routine, with some minor adaptions.
    # What follows is my code: So abandon all hope, you who enter here!

    # get relevant elevation information
    min_hgt, max_hgt = get_min_max_elevation(gdir)

    # get temperature at glacier terminus
    temp_terminus = compute_temp_terminus(itemp, igrad, ref_hgt, min_hgt)
    # compute positive 'melting' temperature/energy input
    temp = np.clip(temp_terminus - temp_melt, a_min=0, a_max=None)
    # get solid precipitation
    prcp_solid = compute_solid_prcp(iprcp, prcp_fac, ref_hgt,
                                    min_hgt, max_hgt,
                                    temp_terminus, temp_all_solid,
                                    igrad, prcp_grad)

    # check if climate data includes all 12 month of all years
    ny, r = divmod(len(time), 12)
    if r != 0:
        raise ValueError('Climate data should be N full years exclusively')
    # last year gives the tone of the hydro year
    years = np.arange(time[-1].year - ny + 1, time[-1].year + 1, 1)

    # compute sums over hydrological year
    temp_yr = np.zeros(len(years))
    prcp_yr = np.zeros(len(years))
    for i, y in enumerate(years):
        temp_yr[i] = np.sum(temp[i * 12:(i + 1) * 12])
        prcp_yr[i] = np.sum(prcp_solid[i * 12:(i + 1) * 12])

    return years, temp_yr, prcp_yr


def _fallback_local_t_star(gdir):
    """A Fallback function if vascaling.local_t_star raises an Error.

    This function will still write a `vascaling_mustar.json`, filled with NANs,
    if vascaling.local_t_star fails and cfg.PARAMS['continue_on_error'] = True.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process

    """
    # Scalars in a small dict for later
    df = dict()
    df['rgi_id'] = gdir.rgi_id
    df['t_star'] = np.nan
    df['bias'] = np.nan
    df['mu_star'] = np.nan
    gdir.write_json(df, 'vascaling_mustar')


@entity_task(log, writes=['vascaling_mustar', 'climate_info'],
             fallback=_fallback_local_t_star)
def local_t_star(gdir, ref_df=None, tstar=None, bias=None):
    """Compute the local t* and associated glacier-wide mu*.

    If `tstar` and `bias` are not provided, they will be interpolated from the
    reference t* list.
    The mass balance calibration parameters (i.e. temperature lapse rate,
    temperature thresholds for melting, solid and liquid precipitation,
    precipitation scaling factor) are written to the climate_info.pkl file.

    The results of the calibration process (i.e. t*, mu*, bias) are stored in
    the `vascaling_mustar.json` file, to be used later by other tasks.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    ref_df : :py:class:`pandas.Dataframe`, optional
        replace the default calibration list with a custom one
    tstar : int, optional
        the year when the glacier should be in equilibrium, default = None
    bias : float, optional
        the associated reference bias, default = None

    """

    if tstar is None or bias is None:
        # Do our own interpolation of t_start for given glacier
        if ref_df is None:
            if not cfg.PARAMS['run_mb_calibration']:
                # Make some checks and use the default one
                climate_info = gdir.get_climate_info()
                source = climate_info['baseline_climate_source']
                ok_source = ['CRU TS4.01', 'CRU TS3.23', 'HISTALP']
                # ok_source = ['HISTALP']
                if not np.any(s in source.upper() for s in ok_source):
                    msg = ('If you are using a custom climate file you should '
                           'run your own MB calibration.')
                    raise MassBalanceCalibrationError(msg)

                # major RGI version relevant
                v = gdir.rgi_version[0]
                # baseline climate
                str_s = 'cru4' if 'CRU' in source else 'histalp'
                # read calibration params reference table
                fn = 'vas_ref_tstars_rgi{}_{}_calib_params.json'.format(v,
                                                                        str_s)
                fp = get_ref_tstars_filepath(fn)
                calib_params = json.load(open(fp))
                for key, value in calib_params.items():
                    if cfg.PARAMS[key] != value:
                        msg = ('The reference t* list you are trying to use '
                               'was calibrated with different MB parameters.')
                        raise MassBalanceCalibrationError(msg)

                # read reference table
                fn = 'vas_ref_tstars_rgi{}_{}.csv'.format(v, str_s)
                fp = get_ref_tstars_filepath(fn)
                ref_df = pd.read_csv(fp)
            else:
                # Use the the local calibration
                fp = os.path.join(cfg.PATHS['working_dir'], 'ref_tstars.csv')
                ref_df = pd.read_csv(fp)

        # Compute the distance to each glacier
        distances = utils.haversine(gdir.cenlon, gdir.cenlat,
                                    ref_df.lon, ref_df.lat)

        # Take the 10 closest
        aso = np.argsort(distances)[0:9]
        amin = ref_df.iloc[aso]
        distances = distances[aso] ** 2

        # If really close no need to divide, else weighted average
        if distances.iloc[0] <= 0.1:
            tstar = amin.tstar.iloc[0]
            bias = amin.bias.iloc[0]
        else:
            tstar = int(np.average(amin.tstar, weights=1. / distances))
            bias = np.average(amin.bias, weights=1. / distances)

    # Add the climate related params to the GlacierDir to make sure
    # other tools cannot fool around without re-calibration
    out = gdir.get_climate_info()
    out['mb_calib_params'] = {k: cfg.PARAMS[k] for k in MB_PARAMS}
    gdir.write_json(out, 'climate_info')

    # We compute the overall mu* here but this is mostly for testing
    # Climate period
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    yr = [tstar - mu_hp, tstar + mu_hp]

    # get monthly climatological values
    # of terminus temperature and solid precipitation
    years, temp, prcp = get_yearly_mb_temp_prcp(gdir, year_range=yr)

    # solve mass balance equation for mu*
    # note: calving is not considered
    mustar = np.mean(prcp) / np.mean(temp)

    # check for a finite result
    if not np.isfinite(mustar):
        raise climate.MassBalanceCalibrationError('{} has a non finite '
                                                  'mu'.format(gdir.rgi_id))

    # Clip the mu
    if not (cfg.PARAMS['min_mu_star'] < mustar < cfg.PARAMS['max_mu_star']):
        raise climate.MassBalanceCalibrationError('mu* out of '
                                                  'specified bounds.')

    # Scalars in a small dict for later
    df = dict()
    df['rgi_id'] = gdir.rgi_id
    df['t_star'] = int(tstar)
    df['bias'] = bias
    df['mu_star'] = mustar
    gdir.write_json(df, 'vascaling_mustar')


@entity_task(log, writes=['vascaling_mustar', 'climate_info'],
             fallback=_fallback_local_t_star)
def mu_star_calibration_from_geodetic_mb(gdir,
                                         ref_mb=None,
                                         ref_period='',
                                         step_height_for_corr=25,
                                         max_height_change_for_corr=3000,
                                         min_mu_star=None,
                                         max_mu_star=None,
                                         ignore_hydro_months=False,
                                         tstar=None,
                                         ref_df=None):
    """TODO:

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ref_mb : float
        the reference mass-balance to match (units: kg m-2 yr-1)
    ref_period : str, default: PARAMS['geodetic_mb_period']
        one of '2000-01-01_2010-01-01', '2010-01-01_2020-01-01',
        '2000-01-01_2020-01-01'. If `ref_mb` is set, this should still match
        the same format but can be any date.
    step_height_for_corr : float, optional, default=25
        TODO:
    max_height_change_for_corr : float, optional, default=3000
        TODO:
    min_mu_star: float, optional
        defaults to cfg.PARAMS['min_mu_star']
    max_mu_star: float, optional
        defaults to cfg.PARAMS['max_mu_star']
    ignore_hydro_months: bool, optional
        do not raise and error if we are not working on calendar years.
    """

    # The t* is needed later to compute the climatological preicipation amount.
    # Hence, the same tstar/ref_df interpolation as for local_t_star is
    # performed hereafter.
    if tstar is None:
        # Do our own interpolation of t_start for given glacier
        if ref_df is None:
            if not cfg.PARAMS['run_mb_calibration']:
                # Make some checks and use the default one
                climate_info = gdir.get_climate_info()
                source = climate_info['baseline_climate_source']
                ok_source = ['CRU TS4.01', 'CRU TS3.23', 'HISTALP']
                # ok_source = ['HISTALP']
                if not np.any(s in source.upper() for s in ok_source):
                    msg = ('If you are using a custom climate file you should '
                           'run your own MB calibration.')
                    raise MassBalanceCalibrationError(msg)

                # major RGI version relevant
                v = gdir.rgi_version[0]
                # baseline climate
                str_s = 'cru4' if 'CRU' in source else 'histalp'
                # read calibration params reference table
                fn = 'vas_ref_tstars_rgi{}_{}_calib_params.json'.format(v,
                                                                        str_s)
                fp = get_ref_tstars_filepath(fn)
                calib_params = json.load(open(fp))
                for key, value in calib_params.items():
                    if cfg.PARAMS[key] != value:
                        msg = ('The reference t* list you are trying to use '
                               'was calibrated with different MB parameters.')
                        raise MassBalanceCalibrationError(msg)

                # read reference table
                fn = 'vas_ref_tstars_rgi{}_{}.csv'.format(v, str_s)
                fp = get_ref_tstars_filepath(fn)
                ref_df = pd.read_csv(fp)
            else:
                # Use the the local calibration
                fp = os.path.join(cfg.PATHS['working_dir'], 'ref_tstars.csv')
                ref_df = pd.read_csv(fp)

        # Compute the distance to each glacier
        distances = utils.haversine(gdir.cenlon, gdir.cenlat,
                                    ref_df.lon, ref_df.lat)

        # Take the 10 closest
        aso = np.argsort(distances)[0:9]
        amin = ref_df.iloc[aso]
        distances = distances[aso] ** 2

        # If really close no need to divide, else weighted average
        if distances.iloc[0] <= 0.1:
            tstar = amin.tstar.iloc[0]
        else:
            tstar = int(np.average(amin.tstar, weights=1. / distances))

    # use default mu* constraints if none are given
    if min_mu_star is None:
        min_mu_star = cfg.PARAMS['min_mu_star']
    if max_mu_star is None:
        max_mu_star = cfg.PARAMS['max_mu_star']

    sm = cfg.PARAMS['hydro_month_' + gdir.hemisphere]
    if sm != 1 and not ignore_hydro_months:
        raise InvalidParamsError('mu_star_calibration_from_geodetic_mb makes '
                                 'more sense when applied on calendar years '
                                 "(PARAMS['hydro_month_nh']=1 and "
                                 "`PARAMS['hydro_month_sh']=1). If you want "
                                 "to ignore this error, set "
                                 "ignore_hydro_months to True")

    # Throw an error if the upper mu* limit is too high
    if max_mu_star > 1000:
        raise InvalidParamsError('You seem to have set a very high '
                                 'max_mu_star for this run. This is not '
                                 'how this task is supposed to work, and '
                                 'we recommend a value lower than 1000 '
                                 '(or even 600).')

    # use default reference period if none is given
    if not ref_period:
        ref_period = cfg.PARAMS['geodetic_mb_period']

    # get start and end year from reference period parameter
    y0, y1 = ref_period.split('_')
    y0 = int(y0.split('-')[0])
    y1 = int(y1.split('-')[0])
    # define year range
    yr_range = [y0, y1 - 1]

    # get yearly climate information
    _, temp, prcp = get_yearly_mb_temp_prcp(gdir, year_range=yr_range)

    if ref_mb is None:
        # get the reference data if not given
        ref_mb = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
        ref_mb = float(ref_mb.loc[ref_mb['period'] == ref_period]['dmdtda'])
        # convert dmdtda from meters water-equivalent per year into kg m-2 yr-1
        ref_mb *= 1000

    def _mu_star_per_minimization(x, ref_mb, temp, prcp):
        return np.mean((prcp - x * temp) - ref_mb)

    try:
        mu_star = brentq(_mu_star_per_minimization,
                         min_mu_star, max_mu_star,
                         args=(ref_mb, temp, prcp),
                         xtol=_brentq_xtol)
    except ValueError:
        # This happens when out of bounds

        # Funny enough, this bias correction is arbitrary.
        # Here I'm trying something arbitrary as well.
        # Let's try to find a range of corrections that would lead to an
        # allowed mu* and pick one

        # Here we ignore the previous QC correction - if any -
        # to ensure that results are the same even after previous correction
        fpath = gdir.get_filepath('climate_historical')
        with utils.ncDataset(fpath, 'a') as nc:
            start = getattr(nc, 'uncorrected_ref_hgt', nc.ref_hgt)
            nc.uncorrected_ref_hgt = start
            nc.ref_hgt = start

        # Read timeseries again after reset
        _, temp, prcp = get_yearly_mb_temp_prcp(gdir, year_range=yr_range)

        # Check in which direction we should correct the temp
        _lim0 = _mu_star_per_minimization(min_mu_star, ref_mb, temp, prcp)
        if _lim0 < 0:
            # The mass-balances are too positive to be matched - we need to
            # cool down the climate data
            step = -step_height_for_corr
            end = -max_height_change_for_corr
        else:
            # The other way around
            step = step_height_for_corr
            end = max_height_change_for_corr

        steps = np.arange(start, start + end, step, dtype=np.int64)
        mu_candidates = steps * np.NaN
        for i, h in enumerate(steps):
            with utils.ncDataset(fpath, 'a') as nc:
                nc.ref_hgt = h

            # Read timeseries
            _, temp, prcp = get_yearly_mb_temp_prcp(gdir, year_range=yr_range)

            try:
                mu_star = brentq(_mu_star_per_minimization,
                                 min_mu_star, max_mu_star,
                                 args=(ref_mb, temp, prcp),
                                 xtol=_brentq_xtol)
            except ValueError:
                mu_star = np.NaN

            # Done - store for later
            mu_candidates[i] = mu_star

        sel_steps = steps[np.isfinite(mu_candidates)]
        sel_mus = mu_candidates[np.isfinite(mu_candidates)]
        if len(sel_mus) == 0:
            # Yeah nothing we can do here
            raise MassBalanceCalibrationError('We could not find a way to '
                                              'correct the climate data and '
                                              'fit within the prescribed '
                                              'bounds for mu*.')

        # Now according to all the corrections we have a series of candidates
        # Her we just pick the first, but to be fair it is arbitrary
        # We could also pick one randomly...
        mu_star = sel_mus[0]
        # Final correction of the data
        with utils.ncDataset(fpath, 'a') as nc:
            nc.ref_hgt = sel_steps[0]
        gdir.add_to_diagnostics('ref_hgt_calib_diff', sel_steps[0] - start)

    if not np.isfinite(mu_star):
        raise MassBalanceCalibrationError('{} '.format(gdir.rgi_id) +
                                          'has a non finite mu.')

    # Add the climate related params to the GlacierDir to make sure
    # other tools cannot fool around without re-calibration
    out = gdir.get_climate_info()
    out['mb_calib_params'] = {k: cfg.PARAMS[k] for k in MB_PARAMS}
    gdir.write_json(out, 'climate_info')

    # Store diagnostics
    df = dict()
    df['rgi_id'] = gdir.rgi_id
    df['t_star'] = t_star
    df['bias'] = 0
    df['mu_star'] = mu_star
    # Write
    gdir.write_json(df, 'vascaling_mustar')


@entity_task(log, writes=['climate_info'])
def t_star_from_refmb(gdir, mbdf=None):
    """Computes the reference year t* for the given glacier and mass balance
    measurements.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    mbdf : :py:class:`pd.Series`
        observed MB data indexed by year. If None, read automatically from the
        reference data, default = None

    Returns
    -------
    dict
        A dictionary {'t_star': [], 'bias': []} containing t* and the
        corresponding mass balance bias


    """
    # make sure we have no marine terminating glacier
    assert gdir.terminus_type == 'Land-terminating'
    # get reference time series of mass balance measurements
    if mbdf is None:
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']

    # compute average observed mass-balance
    ref_mb = np.mean(mbdf)

    # Compute one mu candidate per year and the associated statistics
    # Only get the years were we consider looking for t*
    y0, y1 = cfg.PARAMS['tstar_search_window']
    ci = gdir.get_climate_info()
    y0 = y0 or ci['baseline_hydro_yr_0']
    y1 = y1 or ci['baseline_hydro_yr_1']
    years = np.arange(y0, y1 + 1)

    ny = len(years)
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    mb_per_mu = pd.Series(index=years, dtype=float)

    # get mass balance relevant climate parameters
    years, temp, prcp = get_yearly_mb_temp_prcp(gdir, year_range=[y0, y1])

    # get climate parameters, but only for years with mass balance measurements
    selind = np.searchsorted(years, mbdf.index)
    sel_temp = temp[selind]
    sel_prcp = prcp[selind]
    sel_temp = np.mean(sel_temp)
    sel_prcp = np.mean(sel_prcp)

    # for each year in the climatic period around t* (ignoring the first and
    # last 15-years), compute a mu-candidate by solving the mass balance
    # equation for mu. afterwards compute the average (modeled) mass balance
    # over all years with mass balance measurements using the mu-candidate
    for i, y in enumerate(years):
        # ignore begin and end, i.e. if the
        if ((i - mu_hp) < 0) or ((i + mu_hp) >= ny):
            continue

        # compute average melting temperature
        t_avg = np.mean(temp[i - mu_hp:i + mu_hp + 1])
        # skip if if too cold, i.e. no melt occurs (division by zero)
        if t_avg < 1e-3:
            continue
        # compute the mu candidate for the current year, by solving the mass
        # balance equation for mu*
        mu = np.mean(prcp[i - mu_hp:i + mu_hp + 1]) / t_avg

        # compute mass balance using the calculated mu and the average climate
        # conditions over the years with mass balance records
        mb_per_mu[y] = np.mean(sel_prcp - mu * sel_temp)

    # compute differences between computed mass balance and reference value
    diff = (mb_per_mu - ref_mb).dropna()
    # raise error if no mu could be calculated for any year
    if len(diff) == 0:
        raise MassBalanceCalibrationError('No single valid mu candidate for '
                                          'this glacier!')

    # choose mu* as the mu candidate with the smallest absolute bias
    amin = np.abs(diff).idxmin()

    # write results to the `climate_info.pkl`
    d = gdir.get_climate_info()
    d['t_star'] = amin
    d['bias'] = diff[amin]
    gdir.get_climate_info()

    return {'t_star': amin, 'bias': diff[amin],
            'avg_mb_per_mu': mb_per_mu, 'avg_ref_mb': ref_mb}


@global_task(log)
def compute_ref_t_stars(gdirs):
    """Detects the best t* for the reference glaciers and writes them to disk

    This task will be needed for mass balance calibration of custom climate
    data. For CRU and HISTALP baseline climate a pre-calibrated list is
    available and should be used instead.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        will be filtered for reference glaciers

    """

    if not cfg.PARAMS['run_mb_calibration']:
        raise InvalidParamsError('Are you sure you want to calibrate the '
                                 'reference t*? There is a pre-calibrated '
                                 'version available. If you know what you are '
                                 'doing and still want to calibrate, set the '
                                 '`run_mb_calibration` parameter to `True`.')

    # Reference glaciers only if in the list and period is good
    ref_gdirs = utils.get_ref_mb_glaciers(gdirs)

    # Run
    from oggm.workflow import execute_entity_task
    out = execute_entity_task(t_star_from_refmb, ref_gdirs)

    # Loop write
    df = pd.DataFrame()
    for gdir, res in zip(ref_gdirs, out):
        # list of mus compatibles with refmb
        rid = gdir.rgi_id
        df.loc[rid, 'lon'] = gdir.cenlon
        df.loc[rid, 'lat'] = gdir.cenlat
        df.loc[rid, 'n_mb_years'] = len(gdir.get_ref_mb_data())
        df.loc[rid, 'tstar'] = res['t_star']
        df.loc[rid, 'bias'] = res['bias']

    # Write out
    df['tstar'] = df['tstar'].astype(int)
    df['n_mb_years'] = df['n_mb_years'].astype(int)
    file = os.path.join(cfg.PATHS['working_dir'], 'ref_tstars.csv')
    df.sort_index().to_csv(file)

    # We store the associated params to make sure
    # other tools cannot fool around without re-calibration
    params_file = os.path.join(cfg.PATHS['working_dir'],
                               'vas_ref_tstars_params.json')
    with open(params_file, 'w') as fp:
        json.dump({k: cfg.PARAMS[k] for k in MB_PARAMS}, fp)


@entity_task(log)
def find_start_area(gdir, year_start=1851, adjust_term_elev=False,
                    instant_geometry_change=False):
    """This task find the start area for the given glacier, which results in
    the best results after the model integration (i.e., modeled glacier surface
    closest to measured RGI surface in 2003).

    All necessary prepro task (gis, centerline, climate) must be executed
    beforehand, as well as the local_t_star() task.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    year_start : int, optional
        year at the beginning of the model integration, default = 1851
        (best choice for working with HISTALP data)
    adjust_term_elev: bool, optional, default=False
        flag deciding wheter or not to update the terminus elevation with the
        new initial glacier surface area (not done by Marzeion et al. (2012))

    Returns
    -------
    :py:class:`scipy.optimize.OptimizeResult`

    """

    # instance the mass balance models
    mbmod = VAScalingMassBalance(gdir)

    # get reference area and year from RGI
    a_rgi = gdir.rgi_area_m2
    try:
        y_rgi = gdir.rgi_date.year
    except AttributeError:
        y_rgi = gdir.rgi_date
    # rgi_df = utils.get_rgi_glacier_entities([gdir.rgi_id])
    # y_rgi = int(rgi_df.BgnDate.values[0][:4])
    y_rgi += 1
    # get min and max glacier surface elevation
    h_min, h_max = get_min_max_elevation(gdir)

    # set up the glacier model with the reference values (from RGI)
    model_ref = VAScalingModel(year_0=y_rgi, area_m2_0=a_rgi,
                               min_hgt=h_min, max_hgt=h_max,
                               mb_model=mbmod, glacier_type=gdir.glacier_type)

    def _to_minimize(area_m2_start, ref, _year_start=year_start,
                     _adjust_term_elev=adjust_term_elev):
        """Initialize VAS glacier model as copy of the reference model (ref)
        and adjust the model to the given starting area (area_m2_start) and
        starting year (1851). Let the model evolve to the same year as the
        reference model. Compute and return the relative absolute area error.

        Parameters
        ----------
        area_m2_start : float
        ref : :py:class:`oggm.VAScalingModel`
        _year_start : float, optional
             the default value is inherited from the surrounding task
        adjust_term_elev: bool, optional
            flag deciding wheter or not to update the terminus elevation with
            the new initial glacier surface area

        Returns
        -------
        float
            relative absolute area estimate error

        """
        # define model
        model_tmp = VAScalingModel(year_0=ref.year_0,
                                   area_m2_0=ref.area_m2_0,
                                   min_hgt=ref.min_hgt_0,
                                   max_hgt=ref.max_hgt,
                                   mb_model=ref.mb_model,
                                   glacier_type=ref.glacier_type)
        # scale to desired starting size
        model_tmp.create_start_glacier(area_m2_start, year_start=_year_start,
                                       adjust_term_elev=_adjust_term_elev)
        # run and compare, return relative error
        return np.abs(model_tmp.run_and_compare(ref,
                                                instant_geometry_change=
                                                instant_geometry_change))

    # define bounds - between 100m2 and two times the reference size
    area_m2_bounds = [100, 2 * model_ref.area_m2_0]
    # run minimization
    minimization_res = minimize_scalar(_to_minimize, args=(model_ref),
                                       bounds=area_m2_bounds,
                                       method='bounded')

    return minimization_res


@entity_task(log)
def fixed_geometry_mass_balance(gdir, ys=None, ye=None, years=None,
                                monthly_step=False,
                                climate_filename='climate_historical',
                                climate_input_filesuffix=''):
    """ Re-implementation from the OGGM, see original docstring below:

    Computes the mass-balance with climate input from e.g. CRU or a GCM.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year of the model run (default: from the climate file)
        date)
    ye : int
        end year of the model run (default: from the climate file)
    years : array of ints
        override ys and ye with the years of your choice
    monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    """

    if monthly_step:
        raise NotImplementedError('monthly_step not implemented yet')

    mb = VAScalingMassBalance(gdir, filename=climate_filename,
                              input_filesuffix=climate_input_filesuffix)

    if years is None:
        if ys is None:
            ys = mb.ys
        if ye is None:
            ye = mb.ye
        years = np.arange(ys, ye + 1)

    min_hgt, max_hgt = get_min_max_elevation(gdir)
    odf = pd.Series(data=mb.get_specific_mb(year=years,
                                            min_hgt=min_hgt,
                                            max_hgt=max_hgt),
                    index=years)
    return odf


def compile_fixed_geometry_mass_balance(gdirs, filesuffix='', path=True,
                                        ys=None, ye=None, years=None):
    """ Re-implementation from the OGGM, see original docstring below:

    Compiles a table of specific mass-balance timeseries for all glaciers.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    ys : int
        start year of the model run (default: from the climate file)
        date)
    ye : int
        end year of the model run (default: from the climate file)
    years : array of ints
        override ys and ye with the years of your choice
    """
    # get fixed geometry mass balance for all given glaciers
    out_df = workflow.execute_entity_task(fixed_geometry_mass_balance, gdirs,
                                          ys=ys, ye=ye, years=years)

    # combine into one DataFrame and handle missing data
    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.NaN)
    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    # store to file
    if path:
        if path is True:
            out.to_csv(os.path.join(cfg.PATHS['working_dir'],
                                    ('vas_fixed_geometry_mass_balance' +
                                     filesuffix + '.csv')))
        else:
            out.to_csv(path)
    return out


def match_regional_geodetic_mb(gdirs, rgi_reg=None, dataset='hugonnet',
                               period='2000-01-01_2020-01-01'):
    """ Re-implementation from the OGGM, see original docstring below:

    Regional shift of the mass-balance residual to match observations.
    This is useful for operational runs, but also quite hacky.
    Let's hope we won't need this for too long.

    Parameters
    ----------
    gdirs : the list of gdirs (ideally the entire region)
    rgi_reg : str
       the rgi region to match
    dataset : str
       'hugonnet', or 'zemp'
    period : str
       for 'hugonnet' only. One of
       '2000-01-01_2010-01-01',
       '2010-01-01_2020-01-01',
       '2006-01-01_2019-01-01',
       '2000-01-01_2020-01-01'.
       For 'zemp', the period is always 2006-2016.
    """

    # Get the mass-balance VAS would give out of the box
    df = compile_fixed_geometry_mass_balance(gdirs, path=False)
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    # And also the area
    dfs = utils.compile_glacier_statistics(gdirs, path=False)

    # define start and end year depending on dataset
    if dataset == 'hugonnet':
        y0 = int(period.split('_')[0].split('-')[0])
        y1 = int(period.split('_')[1].split('-')[0]) - 1
    elif dataset == 'zemp':
        y0, y1 = 2006, 2015

    # subset for given period
    odf = pd.DataFrame(df.loc[y0:y1].mean(), columns=['SMB'])
    # add area to dataframe
    odf['AREA'] = dfs.rgi_area_km2 * 1e6

    # Compare area with total RGI area
    rdf = 'rgi62_areas.csv'
    rdf = pd.read_csv(utils.get_demo_file(rdf), dtype={'O1Region': str})
    ref_area = rdf.loc[rdf['O1Region'] == rgi_reg].iloc[0]['AreaNoC2NoNominal']
    diff = (1 - odf['AREA'].sum() * 1e-6 / ref_area) * 100
    msg = 'Applying geodetic MB correction on RGI reg {}. Diff area: {:.2f}%'
    log.workflow(msg.format(rgi_reg, diff))

    # Total MB OGGM
    out_smb = np.average(odf['SMB'], weights=odf['AREA'])  # for logging
    smb_oggm = out_smb

    # Total MB Reference
    if dataset == 'hugonnet':
        df = 'table_hugonnet_regions_10yr_20yr_ar6period.csv'
        df = pd.read_csv(utils.get_demo_file(df))
        df = df.loc[df.period == period].set_index('reg')
        smb_ref = df.loc[int(rgi_reg), 'dmdtda']
    elif dataset == 'zemp':
        df = 'zemp_ref_2006_2016.csv'
        df = pd.read_csv(utils.get_demo_file(df), index_col=0)
        smb_ref = df.loc[int(rgi_reg), 'SMB'] * 1000

    # Total MB Reference
    df = 'table_hugonnet_regions_10yr_20yr_ar6period.csv'
    df = pd.read_csv(utils.get_demo_file(df))
    df = df.loc[df.period == '2006-01-01_2019-01-01'].set_index('reg')
    smb_ref = df.loc[int(rgi_reg), 'dmdtda']

    # Diff between the two
    residual = smb_ref - smb_oggm

    # Let's just shift
    log.workflow('Shifting regional MB bias by {}'.format(residual))
    log.workflow('Observations give {}'.format(smb_ref))
    log.workflow('OGGM SMB gives {}'.format(out_smb))
    for gdir in gdirs:
        try:
            df = gdir.read_json('vascaling_mustar')
            gdir.add_to_diagnostics('mb_bias_before_geodetic_corr', df['bias'])
            df['bias'] = df['bias'] - residual
            gdir.write_json(df, 'vascaling_mustar')
        except FileNotFoundError:
            pass


class VAScalingMassBalance(MassBalanceModel):
    """Original mass balance model, used in Marzeion et. al., 2012.
    The general concept is similar to the oggm.PastMassBalance model.
    Thereby the main difference is that the Volume/Area Scaling mass balance
    model returns only one glacier wide mass balance value per month or year.
    """

    def __init__(self, gdir, mu_star=None, bias=None,
                 filename='climate_historical', input_filesuffix='',
                 repeat=False, ys=None, ye=None, check_calib_params=True):
        """Initialize.

        Parameters
        ----------
        gdir : :py:class:`oggm.GlacierDirectory`
        mu_star : float, optional
            set to the alternative value of mu* you want to use, while
            the default is to use the calibrated value
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *subtracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data
        input_filesuffix : str, optional
            the file suffix of the input climate file, no suffix as default
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way, default=False
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data)
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data)
        check_calib_params : bool
            OGGM will try hard not to use wrongly calibrated mu* by checking
            the parameters used during calibration and the ones you are
            using at run time. If they don't match, it will raise an error.
            Set to False to suppress this check.

        """
        # initalize of oggm.MassBalanceModel
        super(VAScalingMassBalance, self).__init__()

        # read mass balance parameters from file
        if mu_star is None:
            df = gdir.read_json('vascaling_mustar')
            mu_star = df['mu_star']
        if bias is None:
            if cfg.PARAMS['use_bias_for_run']:
                df = gdir.read_json('vascaling_mustar')
                bias = df['bias']
            else:
                bias = 0.
        # set mass balance parameters
        self.mu_star = mu_star
        self.bias = bias

        # set mass balance calibration parameters
        self.t_solid = cfg.PARAMS['temp_all_solid']
        self.t_melt = cfg.PARAMS['temp_melt']
        prcp_fac = cfg.PARAMS['prcp_scaling_factor']
        default_grad = cfg.PARAMS['temp_default_gradient']
        self.prcp_grad = cfg.PARAMS['prcp_default_gradient']

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = gdir.get_climate_info()['mb_calib_params']
            for k, v in mb_calib.items():
                if v != cfg.PARAMS[k]:
                    raise RuntimeError('You seem to use different mass-'
                                       'balance parameters than used for the '
                                       'calibration. '
                                       'Set `check_calib_params=False` '
                                       'to ignore this warning.')

        # set public attributes
        self.temp_bias = 0.
        self.prcp_fac = 1.
        self.repeat = repeat
        self.hemisphere = gdir.hemisphere

        # read climate file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)
        with ncDataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = netCDF4.num2date(time[:], time.units)
            ny, r = divmod(len(time), 12)
            if r != 0:
                raise ValueError('Climate data should be N full years')
            # This is where we switch to hydro float year format
            # Last year gives the tone of the hydro year
            self.years = np.repeat(np.arange(time[-1].year - ny + 1,
                                             time[-1].year + 1), 12)
            self.months = np.tile(np.arange(1, 13), ny)
            # Read timeseries
            self.temp = nc.variables['temp'][:]
            self.prcp = nc.variables['prcp'][:] * prcp_fac
            if 'gradient' in nc.variables:
                grad = nc.variables['gradient'][:]
                # Security for stuff that can happen with local gradients
                g_minmax = cfg.PARAMS['temp_local_gradient_bounds']
                grad = np.where(~np.isfinite(grad), default_grad, grad)
                grad = np.clip(grad, g_minmax[0], g_minmax[1])
            else:
                grad = self.prcp * 0 + default_grad
            self.grad = grad
            self.ref_hgt = nc.ref_hgt
            self.ys = self.years[0] if ys is None else ys
            self.ye = self.years[-1] if ye is None else ye

        # compute climatological precipitation around t*
        # needed later to estimate the volume/length scaling parameter
        t_star = gdir.read_json('vascaling_mustar')['t_star']
        mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
        yr = [t_star - mu_hp, t_star + mu_hp]
        _, _, prcp_clim = get_yearly_mb_temp_prcp(gdir, year_range=yr)
        # convert from [mm we. yr-1] into SI units [m we. yr-1]
        prcp_clim = prcp_clim * 1e-3
        # Marzeion limits the turnover (prcp_clim) to a minimum of 10 mm we./yr
        self.prcp_clim = np.max([10e-3, np.mean(prcp_clim)])

    def get_monthly_climate(self, min_hgt, max_hgt, year):
        """Compute and return monthly positive terminus temperature
        and solid precipitation amount for given month.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation [m asl.]
        max_hgt : float
            maximal glacier surface elevation [m asl.]
        year : float
            floating year, following the hydrological year convention

        Returns
        -------
        [float, float]
            (temp_for_melt) positive terminus temperature [degC] and
            (prcp_solid) solid precipitation amount [kg/m^2]

        """
        # process given time index
        y, m = floatyear_to_date(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        if y < self.ys or y > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(y, self.ys, self.ye))
        pok = np.where((self.years == y) & (self.months == m))[0][0]

        # Read timeseries
        itemp = self.temp[pok] + self.temp_bias
        iprcp = self.prcp[pok] * self.prcp_fac
        igrad = self.grad[pok]

        # compute terminus temperature
        temp_terminus = compute_temp_terminus(itemp, igrad,
                                              self.ref_hgt, min_hgt)
        # compute positive 'melting' temperature/energy input
        temp_for_melt = np.clip(temp_terminus - self.t_melt,
                                a_min=0, a_max=None)
        # compute solid precipitation
        prcp_solid = compute_solid_prcp(iprcp, 1,
                                        self.ref_hgt, min_hgt, max_hgt,
                                        temp_terminus, self.t_solid, igrad,
                                        self.prcp_grad)

        return temp_for_melt, prcp_solid

    def get_monthly_mb(self, min_hgt, max_hgt, year):
        """Compute and return the glacier wide mass balance
        for the given year/month combination.
        Possible mb bias is applied...

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation [m asl.]
        max_hgt : float
            maximal glacier (surface) elevation [m asl.]
        year : float
            floating year and month, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        # get melting temperature and solid precipitation
        temp_for_melt, prcp_solid = self.get_monthly_climate(min_hgt,
                                                             max_hgt,
                                                             year=year)
        # compute mass balance
        mb_month = prcp_solid - self.mu_star * temp_for_melt
        # apply mass balance bias
        mb_month -= self.bias / SEC_IN_YEAR * SEC_IN_MONTH
        # convert into SI units [m_ice/s]
        return mb_month / SEC_IN_MONTH / self.rho

    def get_annual_climate(self, min_hgt, max_hgt, year):
        """Compute and return monthly positive terminus temperature
        and solid precipitation amount for all months
        of the given (hydrological) year.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation [m asl.]
        max_hgt : float
            maximal glacier (surface) elevation [m asl.]
        year : float
            floating year, following the hydrological year convention

        Returns
        -------
        [float array, float array]
            (temp_for_melt) monthly positive terminus temperature [degC] and
            (prcp_solid) monthly solid precipitation amount [kg/m2]

        """
        # process given time index
        year = np.floor(year)
        if self.repeat:
            year = self.ys + (year - self.ys) % (self.ye - self.ys + 1)
        if year < self.ys or year > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(year, self.ys, self.ye))
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError('Year {} not in record'.format(int(year)))

        # Read timeseries
        itemp = self.temp[pok] + self.temp_bias
        iprcp = self.prcp[pok] * self.prcp_fac
        igrad = self.grad[pok]

        # compute terminus temperature
        temp_terminus = compute_temp_terminus(itemp, igrad,
                                              self.ref_hgt, min_hgt)
        # compute positive 'melting' temperature/energy input
        temp_for_melt = np.clip(temp_terminus - self.t_melt,
                                a_min=0, a_max=None)
        # compute solid precipitation
        # prcp factor is set to 1 since it the time series is already corrected
        prcp_solid = compute_solid_prcp(iprcp, 1,
                                        self.ref_hgt, min_hgt, max_hgt,
                                        temp_terminus, self.t_solid, igrad,
                                        self.prcp_grad)

        return temp_for_melt, prcp_solid

    def get_annual_mb(self, min_hgt, max_hgt, year):
        """Compute and return the annual glacier wide mass balance for the
        given year. Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            floating year, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        # get annual mass balance climate
        temp_for_melt, prcp_solid = self.get_annual_climate(min_hgt,
                                                            max_hgt,
                                                            year)
        # compute mass balance
        mb_annual = np.sum(prcp_solid - self.mu_star * temp_for_melt)
        # apply bias and convert into SI units
        return (mb_annual - self.bias) / SEC_IN_YEAR / self.rho

    def get_specific_mb(self, min_hgt, max_hgt, year):
        """Compute and return the annual specific mass balance
        for the given year. Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            float year, using the hydrological year convention

        Returns
        -------
        float
            glacier wide average mass balance, units of millimeter water
            equivalent per year [mm w.e./yr]

        """
        # enables the routine to work on a list of years
        # by calling itself for each given year in the list
        if len(np.atleast_1d(year)) > 1:
            out = [
                self.get_specific_mb(min_hgt=min_hgt, max_hgt=max_hgt, year=yr)
                for yr in year]
            return np.asarray(out)

        # get annual mass balance climate
        temp_for_melt, prcp_solid = self.get_annual_climate(min_hgt,
                                                            max_hgt,
                                                            year)
        # compute mass balance
        mb_annual = np.sum(prcp_solid - self.mu_star * temp_for_melt)
        # apply bias
        return mb_annual - self.bias

    def get_monthly_specific_mb(self, min_hgt=None, max_hgt=None, year=None):
        """Compute and return the monthly specific mass balance
        for the given month. Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float, optional
            glacier terminus elevation [m asl.], default = None
        max_hgt : float, optional
            maximal glacier (surface) elevation [m asl.], default = None
        year : float, optional
            float year and month, using the hydrological year convention,
            default = None

        Returns
        -------
        float
            glacier wide average mass balance, units of millimeter water
            equivalent per months [mm w.e./yr]

        """
        # get annual mass balance climate
        temp_for_melt, prcp_solid = self.get_monthly_climate(min_hgt,
                                                             max_hgt,
                                                             year)
        # compute mass balance
        mb_monthly = np.sum(prcp_solid - self.mu_star * temp_for_melt)
        # apply bias and return
        return mb_monthly - (self.bias / SEC_IN_YEAR * SEC_IN_MONTH)

    def get_ela(self, year=None):
        """The ELA can not be calculated using this mass balance model.

        Parameters
        ----------
        year : float, optional

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError('The equilibrium line altitude can not be ' +
                                  'computed for the `VAScalingMassBalance` ' +
                                  'model.')


@entity_task(log)
def run_from_climate_data(gdir, ys=None, ye=None, min_ys=None, max_ys=None,
                          store_monthly_step=False,
                          climate_filename='climate_historical',
                          climate_input_filesuffix='', output_filesuffix='',
                          init_model_filesuffix=None, init_model_yr=None,
                          init_area_m2=None, bias=None, **kwargs):
    """ Runs a glacier with climate input from e.g. CRU or a GCM.

    This will initialize a :py:class:`oggm-vas.core.VAScalingMassBalance` and
    a :py:class:`oggm-vas.core.VAScalingModel`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year of the model run (default: from the glacier geometry
        date if init_model_filesuffix is None, else init_model_yr)
    ye : int
        end year of the model run (default: last year of the provided
        climate file)
    min_ys : int
        if you want to impose a minimum start year, regardless if the glacier
        inventory date is earlier (e.g. if climate data does not reach).
    max_ys : int
        if you want to impose a maximum start year, regardless if the glacier
        inventory date is later (e.g. if climate data does not reach).
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        for the output file
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be combined
        with `init_model_yr`, overwrites `init_area_m2`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_area_m2: float, optional
        glacier area with which the model is initialized, default is RGI value,
        gets overwriten by init_model_filesuffix
    bias : float
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero
    kwargs : dict
        kwargs for the VAScalingMassBalance and/or VAScalingModel instances
    """

    # Initialize model from previous run if filesuffix is specified
    if init_model_filesuffix is not None:
        # read the given model run and create a dummy model
        fp = gdir.get_filepath('model_diagnostics',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)

        if init_model_yr is None:
            # start with last year of initialization run if not specified
            init_model_yr = fmod.last_yr
        fmod.run_until(init_model_yr)
        ys = init_model_yr
    else:
        fmod = None

    # Take from rgi date if not set yet
    if ys is None:
        try:
            ys = gdir.rgi_date.year
        except AttributeError:
            ys = gdir.rgi_date
        # The RGI timestamp is in calendar date - we convert to hydro date,
        # i.e. 2003 becomes 2004 (so that we don't count the MB year 2003
        # in the simulation)
        ys += 1

    # Final crop
    if min_ys is not None:
        ys = ys if ys > min_ys else min_ys
    if max_ys is not None:
        ys = ys if ys < max_ys else max_ys

    # instance mass balance model
    mb_mod = VAScalingMassBalance(gdir, bias=bias, filename=climate_filename,
                                  input_filesuffix=climate_input_filesuffix,
                                  ys=ys, ye=ye, **kwargs)

    if ye is None:
        # Decide from climate (we can run the last year with data as well)
        ye = mb_mod.ye + 1

    # get needed values from glacier directory
    min_hgt, max_hgt = get_min_max_elevation(gdir)
    if init_area_m2 is None:
        init_area_m2 = gdir.rgi_area_m2

    # instance the model
    model = VAScalingModel(year_0=ys, area_m2_0=init_area_m2,
                           min_hgt=min_hgt, max_hgt=max_hgt,
                           mb_model=mb_mod, glacier_type=gdir.glacier_type)
    if fmod:
        # set initial state accordingly
        model.reset_from_filemodel(fmod)

    # specify where to store model diagnostics
    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)
    # run
    model.run_until_and_store(year_end=ye, diag_path=diag_path)

    return model


@entity_task(log)
def run_historic_from_climate_data(gdir, ys, ye=None,
                                   climate_filename='climate_historical',
                                   climate_input_filesuffix='',
                                   output_filesuffix='',
                                   bias=None, **kwargs):
    """ Runs a glacier with climate input from the given start year ys. Thereby
    the glacier model is initialized so that the glacier area equals the RGI
    area at the RGI date. If the RGI date is before the start year the model
    starts at that year (can be cropped afterwards).

    TODO: this is a quick fix, should be revised at some point

    This will initialize a :py:class:`oggm-vas.core.VAScalingMassBalance` and
    a :py:class:`oggm-vas.core.VAScalingModel`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year of the model run (default: from the glacier geometry
        date if init_model_filesuffix is None, else init_model_yr), get
        overriden by the glacier geometry date if it is before the start year
    ye : int
        end year of the model run (default: last year of the provided
        climate file)
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        for the output file
    bias : float
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero
    kwargs : dict
        kwargs for the VAScalingMassBalance and/or VAScalingModel instances
    """

    # get RGI date
    try:
        rgi_date = gdir.rgi_date.year
    except AttributeError:
        rgi_date = gdir.rgi_date
    # The RGI timestamp is in calendar date - we convert to hydro date,
    # i.e. 2003 becomes 2004 (so that we don't count the MB year 2003
    # in the simulation)
    rgi_date += 1

    # start from RGI date if it is before the desired start year
    if rgi_date < ys:
        ys = rgi_date

    # instance mass balance model
    mb_mod = VAScalingMassBalance(gdir, bias=bias, filename=climate_filename,
                                  input_filesuffix=climate_input_filesuffix,
                                  ys=ys, ye=ye, **kwargs)

    if ye is None:
        # Decide from climate
        ye = mb_mod.ye

    # get needed values from glacier directory
    min_hgt, max_hgt = get_min_max_elevation(gdir)
    # find start area that results in RGI area
    init_area_m2 = find_start_area(gdir, year_start=ys, adjust_term_elev=False,
                                   instant_geometry_change=False)
    if init_area_m2.success:
        # use minimization result as initial area
        init_area_m2 = float(init_area_m2.x)
    else:
        # throw error if minimization was unsuccessful
        raise RuntimeError(f'No start area for {gdir.rgi_id} '
                           f'in {ys} could be found')

    # instance the model
    model = VAScalingModel(year_0=ys, area_m2_0=init_area_m2,
                           min_hgt=min_hgt, max_hgt=max_hgt,
                           mb_model=mb_mod, glacier_type=gdir.glacier_type)

    # specify where to store model diagnostics
    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)
    # run
    model.run_until_and_store(year_end=ye, diag_path=diag_path)

    return model


class RandomVASMassBalance(MassBalanceModel):
    """Random shuffle of all MB years within a given time period.

    This is useful for finding a possible past glacier state or for sensitivity
    experiments.

    Note that this is going to be sensitive to extreme years in certain
    periods, but it is by far more physically reasonable than other
    approaches based on gaussian assumptions.

    Parameters
    ----------

    """

    def __init__(self, gdir, mu_star=None, bias=None,
                 y0=None, halfsize=15, seed=None,
                 filename='climate_historical', input_filesuffix='',
                 all_years=False, unique_samples=False):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value)
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *subtracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        y0 : int, optional, default: tstar
            the year at the center of the period of interest. The default
            is to use tstar as center.
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        seed : int, optional
            Random seed used to initialize the pseudo-random number generator.
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        all_years : bool
            if True, overwrites ``y0`` and ``halfsize`` to use all available
            years.
        unique_samples: bool
            if true, chosen random mass-balance years will only be available
            once per random climate period-length
            if false, every model year will be chosen from the random climate
            period with the same probability
        """

        super(RandomVASMassBalance, self).__init__()
        # initialize the VAS equivalent of the PastMassBalance model over the
        # whole available climate period
        self.mbmod = VAScalingMassBalance(gdir, mu_star=mu_star, bias=bias,
                                          filename=filename,
                                          input_filesuffix=input_filesuffix)

        # get mb model parameters
        self.prcp_clim = self.mbmod.prcp_clim

        # define years of climate period
        if all_years:
            # use full climate period
            self.years = self.mbmod.years
        else:
            if y0 is None:
                # choose t* as center of climate period
                df = gdir.read_json('vascaling_mustar')
                self.y0 = df['t_star']
            else:
                # set y0 as attribute
                self.y0 = y0
            # use 31-year period around given year `y0`
            self.years = np.arange(self.y0 - halfsize, self.y0 + halfsize + 1)
        # define year range and number of years
        self.yr_range = (self.years[0], self.years[-1] + 1)
        self.ny = len(self.years)
        self.hemisphere = gdir.hemisphere

        # define random state
        self.rng = np.random.RandomState(seed)
        self._state_yr = dict()

        # whether or not to sample with or without replacement
        self.unique_samples = unique_samples
        if self.unique_samples:
            self.sampling_years = self.years

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Precipitation factor to apply to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.prcp_fac = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.bias

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        self.mbmod.bias = value

    def get_state_yr(self, year=None):
        """For a given year, get the random year associated to it."""
        year = int(year)
        if year not in self._state_yr:
            if self.unique_samples:
                # --- Sampling without replacement ---
                if self.sampling_years.size == 0:
                    # refill sample pool when all years were picked once
                    self.sampling_years = self.years
                # choose one year which was not used in the current period
                _sample = self.rng.choice(self.sampling_years)
                # write chosen year to dictionary
                self._state_yr[year] = _sample
                # update sample pool: remove the chosen year from it
                self.sampling_years = np.delete(
                    self.sampling_years,
                    np.where(self.sampling_years == _sample))
            else:
                # --- Sampling with replacement ---
                self._state_yr[year] = self.rng.randint(*self.yr_range)
        return self._state_yr[year]

    def get_monthly_mb(self, min_hgt, max_hgt, year=None):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the glacier wide mass balance
        for the given year/month combination.
        Possible mb bias is applied...

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation [m asl.]
        max_hgt : float
            maximal glacier (surface) elevation [m asl.]
        year : float
            floating year and month, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        ryr, m = floatyear_to_date(year)
        ryr = utils.date_to_floatyear(self.get_state_yr(ryr), m)
        return self.mbmod.get_monthly_mb(min_hgt, max_hgt, year=ryr)

    def get_annual_mb(self, min_hgt, max_hgt, year=None):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the annual glacier wide mass balance for the given
        year. Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            floating year, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        ryr = self.get_state_yr(int(year))
        return self.mbmod.get_annual_mb(min_hgt, max_hgt, year=ryr)

    def get_specific_mb(self, min_hgt, max_hgt, year):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the annual specific mass balance for the given year.
        Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            float year, using the hydrological year convention

        Returns
        -------
        float
            glacier wide average mass balance, units of millimeter water
            equivalent per year [mm w.e./yr]

        """
        ryr = self.get_state_yr(int(year))
        return self.mbmod.get_specific_mb(min_hgt, max_hgt, year=ryr)

    def get_ela(self, year=None):
        """The ELA can not be calculated using this mass balance model.

        Parameters
        ----------
        year : float, optional

        Raises
        -------
        NotImplementedError

        """
        ryr = self.get_state_yr(int(year))
        return self.mbmod.get_ela(year=ryr)


@entity_task(log)
def run_random_climate(gdir, nyears=1000, y0=None, halfsize=15,
                       bias=None, seed=None, temperature_bias=None,
                       climate_filename='climate_historical',
                       climate_input_filesuffix='', output_filesuffix='',
                       init_model_filesuffix=None, init_model_yr=None,
                       init_area_m2=None, unique_samples=False, **kwargs):
    """Runs the random mass balance model for a given number of years.

    This initializes a :py:class:`oggm.core.vascaling.RandomVASMassBalance`,
    and runs and stores a :py:class:`oggm.core.vascaling.VAScalingModel` with
    the given mass balance model.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int, optional
        length of the simulation, default = 1000
    y0 : int, optional
        central year of the random climate period. The default is to be
        centred on t*. Default = None
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1),
        default = 15
    bias : float, optional
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero. Default = None
    seed : int
        seed for the random generator. If you ignore this, the runs will be
        different each time. Setting it to a fixed seed across glaciers can
        be usefull if you want to have the same climate years for all of them
    temperature_bias : float, optional
        add a bias to the temperature timeseries, default = None
    climate_filename : str, optional
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str, optional
        filesuffix for the input climate file
    output_filesuffix : str, optional
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be combined
        with `init_model_yr`, overwrites `init_area_m2`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_area_m2: float, optional
        glacier area with which the model is initialized, default is RGI value,
        gets overwriten by init_model_filesuffix
    unique_samples: bool, optional
        if true, chosen random mass-balance years will only be available once
        per random climate period-length
        if false, every model year will be chosen from the random climate
        period with the same probability (default)

    Returns
    -------
    :py:class:`oggm.core.vascaling.VAScalingModel`
    """

    # Initialize model from previous run if filesuffix is specified
    if init_model_filesuffix is not None:
        # read the given model run and create a dummy model
        fp = gdir.get_filepath('model_diagnostics',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)

        if init_model_yr is None:
            # start with last year of initialization run if not specified
            init_model_yr = fmod.last_yr
        fmod.run_until(init_model_yr)
        ys = init_model_yr
    else:
        fmod = None

    # instance mass balance model
    mb_mod = RandomVASMassBalance(gdir, y0=y0, halfsize=halfsize, bias=bias,
                                  seed=seed, filename=climate_filename,
                                  input_filesuffix=climate_input_filesuffix,
                                  unique_samples=unique_samples)

    if temperature_bias is not None:
        # add given temperature bias to mass balance model
        mb_mod.temp_bias = temperature_bias

    # instance the model
    min_hgt, max_hgt = get_min_max_elevation(gdir)
    if init_area_m2 is None:
        init_area_m2 = gdir.rgi_area_m2
    model = VAScalingModel(year_0=0, area_m2_0=init_area_m2,
                           min_hgt=min_hgt, max_hgt=max_hgt,
                           mb_model=mb_mod)
    if fmod:
        # set initial state accordingly
        model.reset_from_filemodel(fmod, y0=0)
    # specify path where to store model diagnostics
    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)
    # run model
    model.run_until_and_store(year_end=nyears, diag_path=diag_path, **kwargs)

    return model


class ConstantVASMassBalance(MassBalanceModel):
    """Constant mass-balance during a chosen period.

    This is useful for equilibrium experiments.

    """

    def __init__(self, gdir, mu_star=None, bias=None,
                 y0=None, halfsize=15, filename='climate_historical',
                 input_filesuffix=''):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value)
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *subtracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        y0 : int, optional, default: tstar
            the year at the center of the period of interest. The default
            is to use tstar as center.
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        """

        super(ConstantVASMassBalance, self).__init__()
        # initialize the VAS equivalent of the PastMassBalance model over the
        # whole available climate period
        self.mbmod = VAScalingMassBalance(gdir, mu_star=mu_star, bias=bias,
                                          filename=filename,
                                          input_filesuffix=input_filesuffix)

        # use t* as the center of the climatological period if not given
        if y0 is None:
            df = gdir.read_json('vascaling_mustar')
            y0 = df['t_star']

        # set model properties
        self.prcp_clim = self.mbmod.prcp_clim
        self.y0 = y0
        self.halfsize = halfsize
        self.years = np.arange(y0 - halfsize, y0 + halfsize + 1)
        self.hemisphere = gdir.hemisphere

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Precipitation factor to apply to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.prcp_fac = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.bias

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        self.mbmod.bias = value

    def get_climate(self, min_hgt, max_hgt, year=None):
        """Average mass balance climate information for given glacier.

        Note that prcp is corrected with the precipitation factor and that
        all other biases (precipitation, temp) are applied.

        Returns
        -------
        [float, float]
            (temp_for_melt) positive terminus temperature [degC] and
            (prcp_solid) solid precipitation amount [kg/m^2]
        """
        # create monthly timeseries over whole climate period
        yrs = utils.monthly_timeseries(self.years[0], self.years[-1],
                                       include_last_year=True)
        # create empty containers
        temp = list()
        prcp = list()
        # iterate over all months
        for i, yr in enumerate(yrs):
            # get positive melting temperature and solid precipitation
            t, p = self.mbmod.get_monthly_climate(min_hgt, max_hgt, year=yr)
            temp.append(t)
            prcp.append(p)
        # Note that we do not weight for number of days per month - bad
        return (np.mean(temp, axis=0),
                np.mean(prcp, axis=0))

    def get_monthly_mb(self, min_hgt, max_hgt, year=None):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the glacier wide mass balance
        for the given year/month combination.
        Possible mb bias is applied...

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation [m asl.]
        max_hgt : float
            maximal glacier (surface) elevation [m asl.]
        year : float
            floating year and month, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        # extract month from year
        _, m = utils.floatyear_to_date(year)
        # sum up the mass balance over all years in climate period
        years = [utils.date_to_floatyear(yr, m) for yr in self.years]
        mb = [self.mbmod.get_annual_mb(min_hgt, max_hgt, year=yr)
              for yr in years]
        # return average value
        return np.average(mb)

    def get_annual_mb(self, min_hgt, max_hgt, year=None):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the annual glacier wide mass balance for the given
        year. Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            floating year, following the hydrological year convention

        Returns
        -------
        float
            average glacier wide mass balance [m/s]

        """
        # sum up the mass balance over all years in climate period
        mb = [self.mbmod.get_annual_mb(min_hgt, max_hgt, year=yr)
              for yr in self.years]
        # return average value
        return np.average(mb)

    def get_specific_mb(self, min_hgt, max_hgt, year=None):
        """ Wrapper around the class intern mass balance model function.
        Compute and return the annual specific mass balance for the given year.
        Possible mb bias is applied.

        Parameters
        ----------
        min_hgt : float
            glacier terminus elevation
        max_hgt : float
            maximal glacier (surface) elevation
        year : float
            float year, using the hydrological year convention

        Returns
        -------
        float
            glacier wide average mass balance, units of millimeter water
            equivalent per year [mm w.e./yr]

        """
        mb = [self.mbmod.get_specific_mb(min_hgt, max_hgt, year=yr)
              for yr in self.years]
        # return average value
        return np.average(mb)

    def get_ela(self, year=None):
        """The ELA can not be calculated using this mass balance model.

        Parameters
        ----------
        year : float, optional

        Raises
        -------
        NotImplementedError

        """
        return self.mbmod.get_ela(year=self.y0)


@entity_task(log)
def run_constant_climate(gdir, nyears=1000, y0=None, halfsize=15,
                         bias=None, temperature_bias=None,
                         climate_filename='climate_historical',
                         climate_input_filesuffix='', output_filesuffix='',
                         init_model_filesuffix=None, init_model_yr=None,
                         init_area_m2=None, **kwargs):
    """
    Runs the constant mass balance model for a given number of years.

    This initializes a :py:class:`oggm.core.vascaling.ConstantVASMassBalance`,
    and runs and stores a :py:class:`oggm.core.vascaling.VAScalingModel` with
    the given mass balance model.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int, optional
        length of the simulation, default = 1000
    y0 : int, optional
        central year of the random climate period. The default is to be
        centred on t*. Default = None
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1),
        default = 15
    bias : float, optional
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero. Default = None
    temperature_bias : float, optional
        add a bias to the temperature timeseries, default = None
    climate_filename : str, optional
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str, optional
        filesuffix for the input climate file
    output_filesuffix : str, optional
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    init_model_filesuffix : str
        if you want to start from a previous model run state. Can be combined
        with `init_model_yr`, overwrites `init_area_m2`
    init_model_yr : int
        the year of the initial run you want to start from. The default
        is to take the last year of the simulation.
    init_area_m2: float, optional
        glacier area with which the model is initialized, default is RGI value,
        gets overwriten by init_model_filesuffix

    Returns
    -------
    :py:class:`oggm.core.vascaling.VAScalingModel`
    """

    # Initialize model from previous run if filesuffix is specified
    if init_model_filesuffix is not None:
        # read the given model run and create a dummy model
        fp = gdir.get_filepath('model_diagnostics',
                               filesuffix=init_model_filesuffix)
        fmod = FileModel(fp)

        if init_model_yr is None:
            # start with last year of initialization run if not specified
            init_model_yr = fmod.last_yr
        fmod.run_until(init_model_yr)
        ys = init_model_yr
    else:
        fmod = None

    # instance mass balance model
    mb_mod = ConstantVASMassBalance(gdir, mu_star=None, bias=bias, y0=y0,
                                    halfsize=halfsize,
                                    filename=climate_filename,
                                    input_filesuffix=climate_input_filesuffix)

    if temperature_bias is not None:
        # add given temperature bias to mass balance model
        mb_mod.temp_bias = temperature_bias

    # instance the model
    min_hgt, max_hgt = get_min_max_elevation(gdir)
    if init_area_m2 is None:
        init_area_m2 = gdir.rgi_area_m2
    model = VAScalingModel(year_0=0, area_m2_0=init_area_m2,
                           min_hgt=min_hgt, max_hgt=max_hgt,
                           mb_model=mb_mod)
    if fmod:
        # set initial state accordingly
        model.reset_from_filemodel(fmod, y0=0)
    # specify path where to store model diagnostics
    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)
    # run model
    model.run_until_and_store(year_end=nyears, diag_path=diag_path, **kwargs)

    return model


class VAScalingModel(object):
    """The volume area scaling glacier model following Marzeion et. al., 2012.

    @TODO: finish DocString

    All used parameters are in SI units (even the climatological precipitation
    (attribute of the mass balance model) is given in [m. we yr-1]).

    Parameters
    ----------

    """

    def __repr__(self):
        """Object representation."""
        return "{}: {}".format(self.__class__, self.__dict__)

    def __str__(self):
        """String representation of the dynamic model, includes current
        year, area, volume, length and terminus elevation."""
        return "{}\nyear: {}\n".format(self.__class__, self.year) \
               + "area [km2]: {:.2f}\n".format(self.area_m2 / 1e6) \
               + "volume [km3]: {:.3f}\n".format(self.volume_m3 / 1e9) \
               + "length [km]: {:.2f}\n".format(self.length_m / 1e3) \
               + "min elev [m asl.]: {:.0f}\n".format(self.min_hgt) \
               + "spec mb [mm w.e. yr-1]: {:.2f}".format(self.spec_mb)

    def __init__(self, year_0, area_m2_0, min_hgt, max_hgt, mb_model,
                 glacier_type='Glacier'):
        """Instance new glacier model.

        year_0: float
            year when the simulation starts
        area_m2_0: float
            starting area at year_0 [m2]
        min_hgt: float
            glacier terminus elevation at year_0 [m asl.]
        max_hgt: float
            maximal glacier surface elevation at year_0 [m asl.]
        mb_model: :py:class:`oggm-vas.VAScalingMassBalance`
            instance of mass balance model
        glacier_type: str, optional, default='Glacier'
            specify whether to use 'Glacier' or 'Ice cap' scaling parameters
        """

        # get constants from cfg.PARAMS
        self.rho = cfg.PARAMS['ice_density']

        # gets scaling parameters depending on the glacier type
        if glacier_type == 'Glacier':
            # get scaling constants
            self.cl = cfg.PARAMS['vas_c_length_m']
            self.ca = cfg.PARAMS['vas_c_area_m2']
            # get scaling exponents
            self.ql = cfg.PARAMS['vas_q_length']
            self.gamma = cfg.PARAMS['vas_gamma_area']
        elif glacier_type == 'Ice cap':
            # get scaling constants
            self.cl = cfg.PARAMS['vas_c_icecap_length_m']
            self.ca = cfg.PARAMS['vas_c_icecap_area_m2']
            # get scaling exponents
            self.ql = cfg.PARAMS['vas_q_icecap_length']
            self.gamma = cfg.PARAMS['vas_gamma_icecap_area']
        else:
            ValueError("Glacier type can only be 'Glacier' or 'Ice cap'.")

        self.glacier_type = glacier_type

        # define temporal index
        self.year_0 = year_0
        self.year = year_0

        # define geometrical/spatial parameters
        self.area_m2_0 = area_m2_0
        self.area_m2 = area_m2_0
        self.min_hgt = min_hgt
        self.min_hgt_0 = min_hgt
        self.max_hgt = max_hgt

        # compute volume (m3) and length (m) from area (using scaling laws)
        self.volume_m3_0 = self.ca * self.area_m2_0 ** self.gamma
        self.volume_m3 = self.volume_m3_0
        # self.length = self.cl * area_0**self.ql
        self.length_m_0 = (self.volume_m3 / self.cl) ** (1 / self.ql)
        self.length_m = self.length_m_0

        # define mass balance model and spec mb
        self.mb_model = mb_model
        self.spec_mb = self.mb_model.get_specific_mb(self.min_hgt,
                                                     self.max_hgt,
                                                     self.year)
        # create geometry change parameters
        self.dL = 0
        self.dA = 0
        self.dV = 0

        # create time scale parameters
        self.tau_a = 1
        self.tau_l = 1

    def _get_specific_mb(self):
        """Invoke `get_specific_mb()` from mass balance model for current year
        and glacier terminus elevation."""
        self.spec_mb = self.mb_model.get_specific_mb(self.min_hgt,
                                                     self.max_hgt,
                                                     self.year)

    def _compute_time_scales(self, factor=1, instant_geometry_change=False):
        """Compute the time scales for glacier length `tau_l`
        and glacier surface area `tau_a` for current time step.
        It is possible to scale the time scales by supplying a multiplicative
        factor, or to simulate instant geometry changes by setting them to 1 yr

        Parameters
        ----------
        factor: int, optional, default=1
        instant_geometry_change: bool, optional, default=False

        """
        if instant_geometry_change or self.volume_m3 == 0 or self.area_m2 == 0:
            # setting the time scales to 1 year can be useful
            self.tau_l = 1
            self.tau_a = 1
        else:
            # compute time scales following Marzeion et al. 2020
            self.tau_l = max(1, (self.volume_m3 / (self.mb_model.prcp_clim
                                                   * self.area_m2)) * factor)
            self.tau_a = max(1, self.tau_l * self.area_m2 / self.length_m ** 2)

    @property
    def volume_km3(self):
        return self.volume_m3 * 1e-9

    @property
    def area_km2(self):
        return self.area_m2 * 1e-6

    @property
    def length_km(self):
        return self.length_m * 1e-3

    def read_from_netcdf(self, path):
        """ Read the model parameters from a model_diagnostics.nc file

        Parameters
        ----------
        path: str
            path to the *.nc file
        """
        with xr.open_dataset(path) as ds:
            self.year_0 = float(ds.time[0].values)

            # get geometrical/spatial parameters
            self.area_m2_0 = float(ds.area_m2[0].values)
            self.area_m2 = float(ds.area_m2[-1].values)
            self.min_hgt = float(ds.min_hgt[-1].values)
            self.min_hgt_0 = float(ds.min_hgt[0].values)
            self.max_hgt = float(ds.max_hgt[0].values)
            self.volume_m3_0 = float(ds.volume_m3[0].values)
            self.volume_m3 = float(ds.volume_m3[-1].values)
            self.length_m_0 = float(ds.length_m[0].values)
            self.length_m = float(ds.length_m[-1].values)

            # get geometry change parameters
            self.dL = float(np.diff(ds.length_m[-2:].values))
            self.dA = float(np.diff(ds.area_m2[-2:].values))
            self.dV = float(np.diff(ds.volume_m3[-2:].values))

            # get specific mass balance
            self.spec_mb = float(ds.spec_mb[-1].values)

            # get time scale parameters
            self.tau_a = float(ds.tau_a[-1].values)
            self.tau_l = float(ds.tau_l[-1].values)

    def reset(self):
        """Set model attributes back to starting values."""
        self.year = self.year_0
        self.length_m = self.length_m_0
        self.area_m2 = self.area_m2_0
        self.volume_m3 = self.volume_m3_0
        self.min_hgt = self.min_hgt_0

        # define mass balance model and spec mb
        self._get_specific_mb()

        # reset geometry change parameters
        self.dL = 0
        self.dA = 0
        self.dV = 0

        # create time scale parameters
        self.tau_a = 1
        self.tau_l = 1

    def reset_year_0(self, y0=None):
        """Set model starting attributes to current values."""
        if y0 is not None:
            self.year_0 = y0
        self.year = self.year_0
        self.length_m_0 = self.length_m
        self.area_m2_0 = self.area_m2
        self.volume_m3_0 = self.volume_m3
        self.min_hgt_0 = self.min_hgt

        # define mass balance model and spec mb
        self._get_specific_mb()

        # reset geometry change parameters
        self.dL = 0
        self.dA = 0
        self.dV = 0

        # create time scale parameters
        self.tau_a = 1
        self.tau_l = 1

    def reset_from_filemodel(self, fmod, y0=None):
        """

        Parameters
        ----------
        fmod

        """
        # get relevant parameters
        self.year = fmod.year

        # define geometrical/spatial parameters
        self.area_m2 = fmod.area_m2
        self.min_hgt = fmod.min_hgt

        # compute volume (m3) and length (m) from area (using scaling laws)
        self.volume_m3 = fmod.volume_m3
        self.length_m = fmod.length_m

        # define mass balance model and spec mb
        self.spec_mb = fmod.spec_mb

        # create time scale parameters
        self.tau_a = fmod.tau_a
        self.tau_l = fmod.tau_l

        # reset initial values
        # self.reset_year_0(y0=self.year if y0 is None else y0)
        if y0 is not None:
            self.year = y0

    def step(self, time_scale_factor=1, instant_geometry_change=False):
        """Advance model glacier by one year. This includes the following:
            - computing time scales
            - computing the specific mass balance
            - computing volume change and new volume
            - computing area change and new area
            - computing length change and new length
            - computing new terminus elevation
        """
        # compute time scales
        self._compute_time_scales(factor=time_scale_factor,
                                  instant_geometry_change=
                                  instant_geometry_change)

        # get specific mass balance B(t)
        self._get_specific_mb()

        # compute volume change dV(t)
        self.dV = self.area_m2 * self.spec_mb / self.rho
        # compute new volume V(t+1)
        self.volume_m3 = max(0, self.volume_m3 + self.dV)

        # compute area change dA(t)
        self.dA = ((self.volume_m3 / self.ca) ** (1 / self.gamma)
                   - self.area_m2) / self.tau_a
        # compute new area A(t+1)
        self.area_m2 = max(0, self.area_m2 + self.dA)
        # compute length change dL(t)
        self.dL = ((self.volume_m3 / self.cl) ** (1 / self.ql)
                   - self.length_m) / self.tau_l
        # compute new length L(t+1)
        self.length_m = max(0, self.length_m + self.dL)
        # compute new terminus elevation min_hgt(t+1)
        self.min_hgt = self.max_hgt + (self.length_m / self.length_m_0
                                       * (self.min_hgt_0 - self.max_hgt))

        # increment year
        self.year += 1

    def run_until(self, year_end, reset=False, time_scale_factor=1,
                  instant_geometry_change=False):
        """Runs the model till the specified year.
        Returns all geometric parameters (i.e. length, area, volume, terminus
        elevation and specific mass balance) at the end of the model evolution.

        Parameters
        ----------
        year_end : float
            end of modeling period
        reset : bool, optional
            If `True`, the model will start from `year_0`, otherwise from its
            current position in time (default).

        Returns
        -------
        [float, float, float, float, float, float]
            the geometric glacier parameters at the end of the model evolution:
            year, length [m], area [m2], volume [m3], terminus elevation
            [m asl.], specific mass balance [mm w.e.]

        """
        # reset parameters to starting values
        if reset:
            self.reset()

        # check validity of end year
        if year_end < self.year:
            # raise warning if model year already past given year, and don't
            # run the model - return current parameters
            raise Warning('Cannot run until {}, already at year {}'.format(
                year_end, self.year))
        else:
            # iterate over all years
            while self.year < year_end:
                # run model for one year
                self.step(time_scale_factor=time_scale_factor,
                          instant_geometry_change=instant_geometry_change)

        # return metrics
        return (self.year, self.length_m, self.area_m2,
                self.volume_m3, self.min_hgt, self.spec_mb)

    def run_until_and_store(self, year_end, diag_path=None,
                            reset=False, time_scale_factor=1,
                            instant_geometry_change=False):
        """Runs the model till the specified year. Returns all relevant
        parameters (i.e. length, area, volume, terminus elevation and specific
        mass balance) for each time step as a xarray.Dataset. If a file path is
        give the dataset is written to file.

        Parameters
        ----------
        year_end : float
            end of modeling period
        run_path : str, optional
            path and filename where to store the model run dataset,
            default = None
        diag_path : str, optional
            path where to store glacier diagnostics, default = None
        reset : bool, optional
            If `True`, the model will start from `year_0`, otherwise from its
            current position in time (default).
        time_scale_factor: int, optional
            linear factor with which to scale the internal time scales,
            default = 1
        instant_geometry_change: bool, optional
            flag deciding whether or not to allow for instant (i.e., yearly)
            geometry changes, neglecting potential response times

        Returns
        -------
        :py:class:`xarray.Dataset`
            model parameters for each time step (year)

        """
        # reset parameters to starting values
        if reset:
            self.reset()

        # check validity of end year
        if year_end < self.year:
            raise ValueError('Cannot run until {}, already at year {}'.format(
                year_end, self.year))

        if not self.mb_model.hemisphere:
            raise InvalidParamsError('run_until_and_store needs a '
                                     'mass-balance model with an unambiguous '
                                     'hemisphere.')

        # define different temporal indices
        yearly_time = np.arange(np.floor(self.year), np.floor(year_end) + 1)

        # TODO: include `store_monthly_step` in parameter list or remove IF:
        store_monthly_step = False
        if store_monthly_step:
            # get monthly time index
            monthly_time = utils.monthly_timeseries(self.year, year_end)
        else:
            # monthly time
            monthly_time = yearly_time.copy()
        # get years and month for hydrological year and calender year
        yrs, months = utils.floatyear_to_date(monthly_time)
        sm = cfg.PARAMS['hydro_month_' + self.mb_model.hemisphere]
        cyrs, cmonths = utils.hydrodate_to_calendardate(yrs, months,
                                                        start_month=sm)

        # get number of temporal indices
        ny = len(yearly_time)
        nm = len(monthly_time)
        # deal with one dimensional temporal indices
        if ny == 1:
            yrs = [yrs]
            cyrs = [cyrs]
            months = [months]
            cmonths = [cmonths]

        # initialize diagnostics output file
        diag_ds = xr.Dataset()

        # Global attributes
        diag_ds.attrs['description'] = 'VAS model output'
        diag_ds.attrs['oggm_version'] = __version__
        diag_ds.attrs['calendar'] = '365-day no leap'
        diag_ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                  gmtime())
        diag_ds.attrs['hemisphere'] = self.mb_model.hemisphere

        # Coordinates
        diag_ds.coords['time'] = ('time', monthly_time)
        diag_ds.coords['hydro_year'] = ('time', yrs)
        diag_ds.coords['hydro_month'] = ('time', months)
        diag_ds.coords['calendar_year'] = ('time', cyrs)
        diag_ds.coords['calendar_month'] = ('time', cmonths)
        # add description as attribute to coordinates
        diag_ds['time'].attrs['description'] = 'Floating hydrological year'
        diag_ds['hydro_year'].attrs['description'] = 'Hydrological year'
        diag_ds['hydro_month'].attrs['description'] = 'Hydrological month'
        diag_ds['calendar_year'].attrs['description'] = 'Calendar year'
        diag_ds['calendar_month'].attrs['description'] = 'Calendar month'

        # create empty variables and attributes
        diag_ds['volume_m3'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['volume_m3'].attrs['description'] = 'Total glacier volume'
        diag_ds['volume_m3'].attrs['unit'] = 'm 3'
        diag_ds['area_m2'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['area_m2'].attrs['description'] = 'Total glacier area'
        diag_ds['area_m2'].attrs['unit'] = 'm 2'
        diag_ds['length_m'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['length_m'].attrs['description'] = 'Glacier length'
        diag_ds['length_m'].attrs['unit'] = 'm 3'
        diag_ds['ela_m'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['ela_m'].attrs['description'] = ('Annual Equilibrium Line '
                                                 'Altitude  (ELA)')
        diag_ds['ela_m'].attrs['unit'] = 'm a.s.l'
        diag_ds['spec_mb'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['spec_mb'].attrs['description'] = 'Specific mass balance'
        diag_ds['spec_mb'].attrs['unit'] = 'mm w.e. yr-1'
        diag_ds['min_hgt'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['min_hgt'].attrs['description'] = 'Terminus elevation'
        diag_ds['min_hgt'].attrs['unit'] = 'm asl.'
        diag_ds['max_hgt'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['max_hgt'].attrs['description'] = 'Maximum surface elevation'
        diag_ds['max_hgt'].attrs['unit'] = 'm asl.'
        diag_ds['tau_l'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['tau_l'].attrs['description'] = 'Length change response time'
        diag_ds['tau_l'].attrs['unit'] = 'years'
        diag_ds['tau_a'] = ('time', np.zeros(nm) * np.NaN)
        diag_ds['tau_a'].attrs['description'] = 'Area change response time'
        diag_ds['tau_a'].attrs['unit'] = 'years'
        # TODO: handel tidewater glaciers
        # TODO: write glacier type and scaling parameters?!

        # run the model
        for i, yr in enumerate(monthly_time):
            self.run_until(yr, time_scale_factor=time_scale_factor,
                           instant_geometry_change=instant_geometry_change)
            # store diagnostics
            diag_ds['volume_m3'].data[i] = self.volume_m3
            diag_ds['area_m2'].data[i] = self.area_m2
            diag_ds['length_m'].data[i] = self.length_m
            diag_ds['spec_mb'].data[i] = self.spec_mb
            diag_ds['min_hgt'].data[i] = self.min_hgt
            diag_ds['max_hgt'].data[i] = self.max_hgt
            diag_ds['tau_l'].data[i] = self.tau_l
            diag_ds['tau_a'].data[i] = self.tau_a

        if diag_path is not None:
            # write to file
            diag_ds.to_netcdf(diag_path)

        return diag_ds

    def run_until_equilibrium(self, rate=0.001, ystep=5, max_ite=200,
                              time_scale_factor=1,
                              instant_geometry_change=False):
        """ Try to run the glacier model until an equilibirum is reached.
        Works only with a constant mass balance model.

        Parameters
        ----------
        rate: float, optional
            rate of volume change for which the glacier is considered to be in
            equilibrium, whereby rate = |V0 - V1| / V0. default is 0.1 percent
        ystep: int, optional
            number of years per iteration step, default is 5
        max_ite: int, optional
            maximum number of iterations, default is 200

        """
        # TODO: isinstance is not working...
        if not isinstance(self.mb_model, ConstantVASMassBalance):
            raise TypeError('The mass balance model must be of type ' +
                            'ConstantVASMassBalance.')
        # initialize the iteration counters and the volume change parameter
        ite = 0
        was_close_zero = 0
        t_rate = 1

        # model runs for a maximum fixed number of iterations
        # loop breaks if an equilibrium is reached (t_rate small enough)
        # or the glacier volume is below 1 for a defined number of times
        while (t_rate > rate) and (ite <= max_ite) and (was_close_zero < 5):
            # increment the iteration counter
            ite += 1
            #  store current volume ('before')
            v_bef = self.volume_m3
            # run for the given number of years
            self.run_until(self.year + ystep,
                           time_scale_factor=time_scale_factor,
                           instant_geometry_change=instant_geometry_change)
            # store new volume ('after')
            v_af = self.volume_m3
            #
            if np.isclose(v_bef, 0., atol=1):
                # avoid division by (values close to) zero
                t_rate = 1
                was_close_zero += 1
            else:
                # compute rate of volume change
                t_rate = np.abs(v_af - v_bef) / v_bef

        # raise RuntimeError if maximum number of iterations is reached
        if ite > max_ite:
            raise RuntimeError('Did not find equilibrium.')

    def create_start_glacier(self, area_m2_start, year_start,
                             adjust_term_elev=False):
        """Instance model with given starting glacier area, for the iterative
        process of seeking the glacier’s surface area at the beginning of the
        model integration.
        Per default, the terminus elevation is not scaled (i.e. is the same as
        for the initial glacier (probably RGI values)). This corresponds to
        the code of Marzeion et. al. (2012), but is physically not consistent.
        It is possible to scale the corresponding terminus elevation given the
        most recent (measured) outline. However, this is not recommended since
        the results may be strange. TODO: this should be fixed sometime...

        Parameters
        ----------
        area_m2_start : float
            starting surface area guess [m2]
        year_start : float
            corresponding starting year
        adjust_term_elev : bool, optional, default = False

        """
        # get terminus elevation from current model
        min_hgt_start = self.min_hgt_0
        # adjust terminus elevation according to new area
        if adjust_term_elev:
            # compute volume (m3) and length (m) from area (using scaling laws)
            volume_m3_start = self.ca * area_m2_start ** self.gamma
            length_m_start = (volume_m3_start / self.cl) ** (1 / self.ql)
            # compute corresponding terminus elevation
            min_hgt_start = self.max_hgt + (length_m_start / self.length_m_0
                                            * (self.min_hgt_0 - self.max_hgt))

        self.__init__(year_start, area_m2_start, min_hgt_start,
                      self.max_hgt, self.mb_model)

    def run_and_compare(self, model_ref, time_scale_factor=1,
                        instant_geometry_change=False):
        """Let the model glacier evolve to the same year as the reference
        model (`model_ref`). Compute and return the relative error in area.

        Parameters
        ----------
        model_ref : :py:class:`oggm.vascaling.VAScalingModel`

        Returns
        -------
        float
            relative surface area error

        """
        # run model and store area
        year, _, area, _, _, _ = self.run_until(year_end=model_ref.year,
                                                reset=True,
                                                time_scale_factor=
                                                time_scale_factor,
                                                instant_geometry_change=
                                                instant_geometry_change)
        assert year == model_ref.year
        # compute relative difference to reference area
        rel_error = 1 - area / model_ref.area_m2

        return rel_error

    def start_area_minimization(start_year):
        """Find the start area which results in a best fitting area after
        model integration.

        """


class FileModel(object):
    """Duck VAS model which actually reads the stuff out of a *.nc file."""

    def __init__(self, path):
        """Instance from file path"""

        ds = xr.open_dataset(path)
        ds.load()

        try:
            self.last_yr = ds.time.values[-1]
        except AttributeError:
            err_msg = 'The provided model output file is incomplete (likely ' \
                      'when the previous run failed) or corrupt.'
            raise oggm.exceptions.InvalidWorkflowError(err_msg)

        self.ds = ds

        self.year_0 = float(ds.time[0].values)

        # get initial geometrical/spatial parameters
        self.length_m_0 = float(ds.length_m[0].values)
        self.area_m2_0 = float(ds.area_m2[0].values)
        self.volume_m3_0 = float(ds.volume_m3[0].values)
        self.min_hgt_0 = float(ds.min_hgt[0].values)
        self.max_hgt = float(ds.max_hgt[0].values)

        # set yearly values to initial values
        self.year = self.year_0
        self.length_m = self.length_m_0
        self.area_m2 = self.area_m2_0
        self.volume_m3 = self.volume_m3_0
        self.min_hgt = self.min_hgt_0

        # define mass balance model and spec mb
        self.spec_mb = float(self.ds.spec_mb[0].values)

        # reset geometry change parameters
        self.dL = 0
        self.dA = 0
        self.dV = 0

        # create time scale parameters
        self.tau_a = 1
        self.tau_l = 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.ds.close()

    def reset(self):
        """Set model attributes back to starting values."""
        self.year = self.year_0
        self.length_m = self.length_m_0
        self.area_m2 = self.area_m2_0
        self.volume_m3 = self.volume_m3_0
        self.min_hgt = self.min_hgt_0

        # define mass balance model and spec mb
        self.spec_mb = float(self.ds.spec_mb[0].values)

        # reset geometry change parameters
        # self.dL = 0
        # self.dA = 0
        # self.dV = 0

        # create time scale parameters
        self.tau_a = 1
        self.tau_l = 1

    def reset_year_0(self, y0=None):
        """Reset the initial model to the given year time"""
        if y0 is None:
            # if no year is given, fallback to self.reset()
            self.reset()
            return

        # get values from given year
        self.run_until(y0)
        # define current year and state as initial state
        self.year_0 = self.y0
        self.length_m_0 = self.length_m
        self.area_m2_0 = self.area_m2
        self.volume_m3_0 = self.volume_m3
        self.min_hgt_0 = self.min_hgt
        self.tau_a = 1
        self.tau_l = 1

    @property
    def volume_km3(self):
        return self.volume_m3 * 1e-9

    @property
    def area_km2(self):
        return self.area_m2 * 1e-6

    @property
    def length_km(self):
        return self.length_m * 1e-3

    def run_until(self, year=None, month=None):
        """Mimics the model's behavior by reading the values of the given year
        from the *.nc file. """
        # adjust date according to the hydrological floating year convention
        if month is not None:
            year += (month - 1) / 12
        # select given date from the *.nc file
        ds_sel = self.ds.sel(time=year)

        # get relevant parameters
        self.year = float(ds_sel.time)

        # define geometrical/spatial parameters
        self.area_m2 = float(ds_sel.area_m2.values)
        self.min_hgt = float(ds_sel.min_hgt.values)

        # compute volume (m3) and length (m) from area (using scaling laws)
        self.volume_m3 = float(ds_sel.volume_m3.values)
        self.length_m = float(ds_sel.length_m.values)

        # define mass balance model and spec mb
        self.spec_mb = float(ds_sel.spec_mb.values)

        # create time scale parameters
        self.tau_a = float(ds_sel.tau_a.values)
        self.tau_l = float(ds_sel.tau_l.values)
