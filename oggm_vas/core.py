""" Implementation of the 'original' volume/area scaling glacier model from
Marzeion et. al. 2012, see http://www.the-cryosphere.net/6/1295/2012/.
While the mass balance model is comparable to OGGMs monthly temperature index
model, the 'dynamic' part does not include any ice physics but works with
area/volume and length/volume scaling instead.

Author: Moritz Oberrauch
"""
# Built ins
import logging
from time import gmtime, strftime

# External libs
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import brentq
from sklearn.linear_model import LinearRegression

# import OGGM modules
import oggm
import oggm.cfg as cfg
from oggm import __version__
from oggm import utils, entity_task, workflow
from oggm.utils import floatyear_to_date, date_to_floatyear, lazy_property
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

from oggm.core import massbalance
from oggm.core.massbalance import MassBalanceModel, MonthlyTIModel

# Module logger
log = logging.getLogger(__name__)

# Diagnostic variables that `oggm.utils.compile_run_output` accepts. The VAS
# specific ones (spec_mb, tau_l, ...) are written to `vas_diagnostics`.
OGGM_DIAG_VARS = ['volume_m3', 'area_m2', 'length_m']


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


def compute_temp_terminus(temp, temp_grad, ref_hgt,
                          terminus_hgt, temp_anomaly=0):
    """Computes the (monthly) mean temperature at the glacier terminus,
    following section 2.1.2 of Marzeion et. al., 2012. The input temperature
    is scaled by the given temperature gradient and the elevation difference
    between reference altitude and the glacier terminus elevation.

    Parameters
    ----------
    temp : np.ndarray
        monthly mean climatological temperature (degC)
    temp_grad : np.ndarray or float
        temperature lapse rate [degC per m of elevation change]
    ref_hgt : float
        reference elevation for climatological temperature [m asl.]
    terminus_hgt : float
        elevation of the glacier terminus (m asl.)
    temp_anomaly : np.ndarray or float, optional
        monthly mean temperature anomaly, default 0

    Returns
    -------
    np.ndarray
        monthly mean temperature at the glacier terminus [degC]

    """
    temp_terminus = temp + temp_grad * (terminus_hgt - ref_hgt) + temp_anomaly
    return temp_terminus


def compute_solid_prcp(prcp, min_hgt, max_hgt, temp_terminus,
                       temp_all_solid, temp_grad):
    """Compute the (monthly) amount of solid precipitation onto the glacier
    surface, following section 2.1.1 of Marzeion et. al., 2012. The fraction of
    solid precipitation depends on the terminus temperature, the temperature
    threshold for solid precipitation and the glacier elevation range.

    Note that - unlike the original Marzeion et al. (2012) model - no
    precipitation lapse rate is applied. Precipitation is used at the
    elevation of the climate file (`ref_hgt`), the same way OGGM does it.

    Parameters
    ----------
    prcp : np.ndarray
        monthly mean climatological precipitation [kg/m2], already scaled
        by the precipitation factor
    min_hgt : float
        minimum glacier elevation [m asl.]
    max_hgt : float
        maximum glacier elevation [m asl.]
    temp_terminus : np.ndarray
        monthly mean temperature at the glacier terminus [degC]
    temp_all_solid : float
        temperature threshold below which all precipitation is solid [degC]
    temp_grad : np.ndarray or float
        temperature lapse rate [degC per m of elevation change]

    Returns
    -------
    np.ndarray
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

    return prcp * f_solid


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
    with utils.ncDataset(fpath) as nc:
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
    rgi_id = [gs.get('rgi_id', np.nan) for gs in glacier_stats]
    length = [gs.get('longuest_centerline_km', np.nan) * 1e3
              for gs in glacier_stats]
    area = [gs.get('rgi_area_km2', np.nan) * 1e6 for gs in glacier_stats]
    volume = [gs.get('inv_volume_km3', np.nan) * 1e9 for gs in glacier_stats]
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
    rgi_id = [gs.get('rgi_id', np.nan) for gs in glacier_stats]
    length = [gs.get('longuest_centerline_km', np.nan) * 1e3
              for gs in glacier_stats]
    area = [gs.get('rgi_area_km2', np.nan) * 1e6 for gs in glacier_stats]
    volume = [gs.get('inv_volume_km3', np.nan) * 1e9 for gs in glacier_stats]
    glacier_type_list = [gs.get('glacier_type', np.nan) for gs in
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


class VAScalingMassBalance(MonthlyTIModel):
    """Mass balance model used in Marzeion et. al., 2012.

    This is a thin specialisation of OGGM's `MonthlyTIModel`: the climate
    handling, the calibrated parameters (`melt_f`, `temp_bias`, `prcp_fac`)
    and the mass balance equation are all inherited. The one thing that
    differs is *where* the climate is evaluated.

    The volume/area scaling model has no elevation profile. It only knows the
    glacier terminus elevation `min_hgt` and the maximum surface elevation
    `max_hgt`. Melt is computed from the temperature at the terminus, and the
    solid fraction of the precipitation from the `min_hgt` / `max_hgt` range
    (Marzeion et. al., 2012, sections 2.1.1 and 2.1.2). The model therefore
    returns one glacier wide value per month or year, and the `heights`
    argument of the inherited methods is ignored.

    Because it honours the same parameters and the same interface as
    `MonthlyTIModel`, it can be handed to OGGM's calibration tasks as
    `mb_model_class=VAScalingMassBalance`.
    """

    def __init__(self, gdir, min_hgt=None, max_hgt=None,
                 prcp_clim_period=None, **kwargs):
        """Initialize.

        Parameters
        ----------
        gdir : :py:class:`oggm.GlacierDirectory`
        min_hgt : float, optional
            glacier terminus elevation [m asl.]. Defaults to the minimum
            glacier surface elevation of the RGI outline.
        max_hgt : float, optional
            maximum glacier surface elevation [m asl.]. Defaults to the
            maximum glacier surface elevation of the RGI outline.
        prcp_clim_period : str, optional
            the period over which to average the climatological solid
            precipitation used for the response time scales, e.g.
            '2000-01-01_2020-01-01'. Defaults to the whole climate record,
            see `prcp_clim`.
        **kwargs
            passed to `oggm.core.massbalance.MonthlyTIModel`, i.e. `melt_f`,
            `temp_bias`, `prcp_fac`, `filename`, `input_filesuffix`,
            `settings_filesuffix`, `bias`, `ys`, `ye`, `repeat`,
            `check_calib_params`, ...

        """
        if min_hgt is None or max_hgt is None:
            _min, _max = get_min_max_elevation(gdir)
            min_hgt = _min if min_hgt is None else min_hgt
            max_hgt = _max if max_hgt is None else max_hgt

        # the terminus moves during a run, the initial value does not
        self.min_hgt_0 = min_hgt
        self.min_hgt = min_hgt
        self.max_hgt = max_hgt
        self.prcp_clim_period = prcp_clim_period

        super(VAScalingMassBalance, self).__init__(gdir, **kwargs)

    def _get_climate_for_index(self, heights, pok):
        """Glacier wide climate for the given time index or indices.

        Overrides `MonthlyTIModel._get_climate_for_index`. `heights` is
        ignored: this model has no elevation profile and returns a single
        "height bin" representing the glacier as a whole. Everything built on
        top of this method (`get_monthly_climate`, `get_annual_climate`,
        `get_monthly_mb`, `get_annual_mb`, ...) is inherited unchanged.

        Parameters
        ----------
        heights : array_like
            ignored
        pok : int or np.ndarray
            the time indices of interest

        Returns
        -------
        tuple[np.ndarray]
            temperature, melt temperature, total and solid precipitation,
            of shape (1, ) for a single time index and (1, n) otherwise.

        """
        # already corrected for temp_bias and prcp_fac by MonthlyTIModel
        # one single "height bin", so 1D for one time index, 2D otherwise
        shaper = np.atleast_1d if np.size(pok) == 1 else np.atleast_2d

        itemp = shaper(np.asarray(self.temp[pok], dtype=np.float64))
        prcp = shaper(np.asarray(self.prcp[pok], dtype=np.float64))
        igrad = shaper(np.asarray(self.grad[pok], dtype=np.float64))

        # temperature at the glacier terminus
        temp = compute_temp_terminus(itemp, igrad, self.ref_hgt, self.min_hgt)
        tempformelt = self._get_tempformelt(temp)

        # solid precipitation onto the glacier surface
        prcpsol = compute_solid_prcp(prcp, self.min_hgt, self.max_hgt,
                                     temp, self.temp_all_solid, igrad)

        return temp, tempformelt, prcp, prcpsol

    def get_specific_mb(self, heights=None, widths=None, fls=None, year=None,
                        time_resolution='annual'):
        """Glacier wide specific mass balance, in mm w.e. per year/month.

        `heights`, `widths` and `fls` are ignored - the geometry this model
        uses is its own `min_hgt` / `max_hgt` pair. They are part of the
        signature so that OGGM's calibration tasks can call this model.
        """
        if fls is not None and len(fls) > 1:
            raise InvalidWorkflowError(
                'The volume/area scaling model describes the glacier as a '
                'whole and cannot be used with more than one flowline. Use '
                'elevation band flowlines (a single flowline per glacier).')

        return super(VAScalingMassBalance, self).get_specific_mb(
            heights=np.array([self.min_hgt]), widths=np.array([1.]),
            fls=None, year=year, time_resolution=time_resolution)

    @lazy_property
    def prcp_clim(self):
        """Climatological solid precipitation onto the glacier [m w.e. yr-1].

        This is the "turnover" used by `VAScalingModel` to estimate the
        glacier response time scales. Marzeion et. al. (2012) averaged it over
        the 31 years centred on t*, which no longer exists in OGGM.

        It is averaged over the whole climate record instead. Tying it to the
        calibration period would make it depend on a value the calibration
        itself writes, so the glacier would evolve differently during and
        after a dynamic calibration. The response time is a property of the
        glacier and its climate rather than of the observation period, and the
        full record samples it better. Pass `prcp_clim_period` to override.

        It is evaluated at the RGI date geometry and, as in the original
        model, limited to a minimum of 10 mm w.e. yr-1.
        """
        if self.prcp_clim_period is None:
            years = np.unique(self.years)
        else:
            y0, y1 = [int(y.split('-')[0])
                      for y in self.prcp_clim_period.split('_')]
            years = [y for y in np.arange(y0, y1) if self.is_year_valid(y)]
            if not years:
                raise InvalidWorkflowError(
                    f'{self.gdir.rgi_id}: the period '
                    f'{self.prcp_clim_period} is not covered by the climate '
                    f'data [{self.ys_float}, {self.ye_float}].')

        # the turnover is a climatology: use the RGI date geometry, whatever
        # the terminus is doing at the moment
        min_hgt = self.min_hgt
        try:
            self.min_hgt = self.min_hgt_0
            prcp_sol = np.array([self._get_2d_annual_climate(None, y)[3].sum()
                                 for y in years])
        finally:
            self.min_hgt = min_hgt

        # convert from [kg m-2 yr-1] into SI units [m we. yr-1]
        return np.max([10e-3, np.mean(prcp_sol) * 1e-3])

    def get_ela(self, year=None, **kwargs):
        """The ELA cannot be computed by this mass balance model."""
        raise NotImplementedError('The equilibrium line altitude can not be '
                                  'computed for the `VAScalingMassBalance` '
                                  'model.')


@entity_task(log, writes=['mb_calib'])
def mb_calibration_from_geodetic_mb(gdir, **kwargs):
    """Calibrate the VAS mass balance model on geodetic mass balance data.

    Convenience wrapper around
    :py:func:`oggm.core.massbalance.mb_calibration_from_geodetic_mb`, which
    does all the work. The only thing this adds is the VAS mass balance model
    as `mb_model_class` and OGGM's default three step calibration order.

    Like OGGM, this writes `melt_f`, `prcp_fac` and `temp_bias` to the
    glacier directory settings.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    **kwargs
        passed to OGGM's `mb_calibration_from_geodetic_mb`

    Returns
    -------
    dict
        the calibrated parameters

    """
    kwargs.setdefault('mb_model_class', VAScalingMassBalance)
    kwargs.setdefault('calibrate_param1', 'melt_f')
    kwargs.setdefault('calibrate_param2', 'prcp_fac')
    kwargs.setdefault('calibrate_param3', 'temp_bias')
    return massbalance.mb_calibration_from_geodetic_mb.unwrapped(gdir, **kwargs)


def _rgi_year(gdir):
    """The RGI inventory year of a glacier directory."""
    try:
        return gdir.rgi_date.year
    except AttributeError:
        return gdir.rgi_date


def _target_year(gdir, mb_model, target_yr=None):
    """The year at which the model geometry is matched against the RGI.

    Like OGGM, the RGI outline is taken to describe the glacier at the start
    of the year following the inventory date.
    """
    if target_yr is None:
        target_yr = _rgi_year(gdir) + 1
    if target_yr > mb_model.ye + 1:
        log.warning('(%s) the RGI date (%d) is not covered by the climate '
                    'data, matching the geometry in %d instead.',
                    gdir.rgi_id, target_yr - 1, mb_model.ye)
        target_yr = mb_model.ye + 1
    return target_yr


def _reference_model(gdir, mb_model, target_yr):
    """A `VAScalingModel` of the observed (RGI) glacier at `target_yr`."""
    min_hgt, max_hgt = get_min_max_elevation(gdir)
    return VAScalingModel(year_0=target_yr, area_m2_0=gdir.rgi_area_m2,
                          min_hgt=min_hgt, max_hgt=max_hgt,
                          mb_model=mb_model,
                          glacier_type=gdir.glacier_type)


def _start_model(model_ref, area_m2_start, year_start,
                 adjust_term_elev=False):
    """A copy of `model_ref` rescaled to `area_m2_start` at `year_start`."""
    model = VAScalingModel(year_0=model_ref.year_0,
                           area_m2_0=model_ref.area_m2_0,
                           min_hgt=model_ref.min_hgt_0,
                           max_hgt=model_ref.max_hgt,
                           mb_model=model_ref.mb_model,
                           glacier_type=model_ref.glacier_type)
    model.create_start_glacier(area_m2_start, year_start=year_start,
                               adjust_term_elev=adjust_term_elev)
    return model


def find_start_area_from_model(model_ref, year_start, target_yr=None,
                               adjust_term_elev=False,
                               instant_geometry_change=False,
                               max_area_factor=100, rtol=1e-4):
    """Find the glacier area at `year_start` that reproduces the reference area.

    The modelled area at `target_yr` grows monotonically with the area the
    glacier started from, so this is a root finding problem, not a
    minimisation. The bracket is widened until it contains the solution.
    That matters for early start years and strongly negative mass balances: a
    glacier that has since lost most of its mass needs a start area many times
    its present one, and a fixed bracket silently returns its own bound.

    Parameters
    ----------
    model_ref : :py:class:`oggm_vas.VAScalingModel`
        the reference model, i.e. the observed glacier at `target_yr`
    year_start : int
        the year to start the reconstruction from
    target_yr : int, optional
        the year at which to match the reference area. Defaults to the
        reference model's own year.
    adjust_term_elev : bool, optional
        whether to move the terminus elevation with the start area. Marzeion
        et al. (2012) do not, which is the default. Note that with the default
        the reconstructed glacier matches the RGI area at `target_yr` but sits
        on a terminus elevation that is higher than the observed one.
    instant_geometry_change : bool, optional
        whether to neglect the response time scales
    max_area_factor : float, optional
        the largest start area to consider, as a multiple of the reference
        area. Raises rather than returning a silent non-match.
    rtol : float, optional
        the relative area error that counts as a match

    Returns
    -------
    float
        the glacier surface area at `year_start` [m2]

    """
    if target_yr is None:
        target_yr = model_ref.year_0
    if year_start >= target_yr:
        raise InvalidParamsError(f'year_start ({year_start}) must be before '
                                 f'the target year ({target_yr}).')

    def _rel_area_error(area_m2_start):
        """Signed relative area error at the target year. Decreases
        monotonically with the start area."""
        model = _start_model(model_ref, area_m2_start, year_start,
                             adjust_term_elev=adjust_term_elev)
        return model.run_and_compare(
            model_ref, instant_geometry_change=instant_geometry_change)

    # a glacier this small cannot end up bigger than the reference one
    area_lo = 100.
    err_lo = _rel_area_error(area_lo)
    if err_lo < 0:
        raise RuntimeError(
            f'Even a {area_lo:.0f} m2 glacier in {year_start} ends up larger '
            f'than the reference area in {target_yr}: the mass balance is too '
            f'positive for a reconstruction.')

    # widen the bracket until it contains the solution
    area_hi = model_ref.area_m2_0
    err_hi = _rel_area_error(area_hi)
    factor = 2.
    while err_hi > 0:
        if factor > max_area_factor:
            raise RuntimeError(
                f'No start area below {max_area_factor:.0f} x the reference '
                f'area reproduces the {target_yr} area from {year_start} '
                f'(best relative area error {err_hi:.3f}). The mass balance '
                'is likely too negative, or the start year too early.')
        area_hi = factor * model_ref.area_m2_0
        err_hi = _rel_area_error(area_hi)
        factor *= 2.

    area_start = brentq(_rel_area_error, area_lo, area_hi, xtol=1.)

    err = abs(_rel_area_error(area_start))
    if err > rtol:
        raise RuntimeError(f'The reconstructed area in {target_yr} is off by '
                           f'{err:.3e} (relative), more than the requested '
                           f'{rtol:.3e}.')

    return area_start


@entity_task(log)
def find_start_area(gdir, year_start=1851, target_yr=None,
                    adjust_term_elev=False, instant_geometry_change=False,
                    max_area_factor=100, rtol=1e-4, **kwargs):
    """Find the glacier area at `year_start` that reproduces the RGI area.

    The glacier is integrated from `year_start` to the RGI date and the start
    area is adjusted until the modelled area matches the inventory. All
    preprocessing tasks (gis, climate) and the mass balance calibration must
    have run beforehand.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    year_start : int, optional
        year at the beginning of the reconstruction, default = 1851
    target_yr : int, optional
        the year at which to match the RGI area. Defaults to the year after
        the inventory date, clipped to the end of the climate record.
    adjust_term_elev : bool, optional
        whether to move the terminus elevation with the start area,
        default = False (as in Marzeion et al., 2012)
    instant_geometry_change : bool, optional
        whether to neglect the response time scales, default = False
    max_area_factor : float, optional
        the largest start area to consider, as a multiple of the RGI area
    rtol : float, optional
        the relative area error that counts as a match
    **kwargs
        passed to :py:class:`VAScalingMassBalance`

    Returns
    -------
    float
        the glacier surface area at `year_start` [m2]

    """
    mbmod = VAScalingMassBalance(gdir, **kwargs)
    target_yr = _target_year(gdir, mbmod, target_yr)
    model_ref = _reference_model(gdir, mbmod, target_yr)
    return find_start_area_from_model(
        model_ref, year_start, target_yr=target_yr,
        adjust_term_elev=adjust_term_elev,
        instant_geometry_change=instant_geometry_change,
        max_area_factor=max_area_factor, rtol=rtol)


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

    odf = pd.Series(data=mb.get_specific_mb(year=years), index=years)
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
            out_df[idx] = pd.Series(np.nan)
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



@entity_task(log)
def run_from_climate_data(gdir, ys=None, ye=None, min_ys=None, max_ys=None,
                          store_monthly_step=False,
                          climate_filename='climate_historical',
                          climate_input_filesuffix='', output_filesuffix='',
                          init_model_filesuffix=None, init_model_yr=None,
                          init_area_m2=None, bias=0, **kwargs):
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
        fp = gdir.get_filepath('vas_diagnostics',
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
        # Start the year after the RGI date, so that we don't count the
        # MB year of the inventory date in the simulation (as OGGM does)
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
    vas_diag_path = gdir.get_filepath('vas_diagnostics',
                                      filesuffix=output_filesuffix,
                                      delete=True)
    # run
    model.run_until_and_store(year_end=ye, diag_path=diag_path,
                              vas_diag_path=vas_diag_path)

    return model


@entity_task(log, writes=['model_diagnostics', 'vas_diagnostics'])
def run_reconstruction(gdir, ys=None, ye=None, target_yr=None,
                       climate_filename='climate_historical',
                       climate_input_filesuffix='', output_filesuffix='',
                       bias=0, adjust_term_elev=False,
                       instant_geometry_change=False,
                       max_area_factor=100, rtol=1e-4, **kwargs):
    """Reconstruct the glacier over the whole available climate record.

    This is one continuous run that passes through the observed geometry: the
    area at `ys` is chosen so that the model reproduces the RGI area at the
    inventory date, and the run then carries on to `ye`. Contrast with
    :py:func:`run_from_climate_data`, which simply imposes the RGI area at
    `ys` whatever year that is - fine when `ys` is the inventory date, wrong
    otherwise.

    Since the inventory date varies from the 1960s to the 2010s across the
    RGI, this is what makes runs comparable between glaciers: they all pass
    through their own observed area at their own observed date.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    ys : int, optional
        start year of the reconstruction. Defaults to the first year of the
        climate record.
    ye : int, optional
        end year of the run. Defaults to the last year of the climate record.
    target_yr : int, optional
        the year at which to match the RGI area. Defaults to the year after
        the inventory date.
    climate_filename : str, optional
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix : str, optional
        filesuffix for the input climate file
    output_filesuffix : str, optional
        for the output file
    bias : float, optional
        the mass balance bias to apply, default 0
    adjust_term_elev : bool, optional
        whether to move the terminus elevation with the start area,
        default = False (as in Marzeion et al., 2012)
    instant_geometry_change : bool, optional
        whether to neglect the response time scales, default = False
    max_area_factor : float, optional
        the largest start area to consider, as a multiple of the RGI area
    rtol : float, optional
        the relative area error that counts as a match
    **kwargs
        passed to :py:class:`VAScalingMassBalance`

    Returns
    -------
    :py:class:`oggm_vas.VAScalingModel`

    """
    mbmod = VAScalingMassBalance(gdir, filename=climate_filename,
                                 input_filesuffix=climate_input_filesuffix,
                                 bias=bias, **kwargs)
    if ys is None:
        ys = int(mbmod.ys)
    if ye is None:
        # we can run the last year with data as well
        ye = int(mbmod.ye) + 1

    target_yr = _target_year(gdir, mbmod, target_yr)
    if ys >= target_yr:
        raise InvalidParamsError(
            f'{gdir.rgi_id}: cannot reconstruct from {ys}, which is not '
            f'before the date the geometry is matched at ({target_yr}).')
    if ye < target_yr:
        raise InvalidParamsError(
            f'{gdir.rgi_id}: the run ends in {ye}, before the date the '
            f'geometry is matched at ({target_yr}).')

    # find the start area that reproduces the observed one
    model_ref = _reference_model(gdir, mbmod, target_yr)
    area_m2_start = find_start_area_from_model(
        model_ref, ys, target_yr=target_yr,
        adjust_term_elev=adjust_term_elev,
        instant_geometry_change=instant_geometry_change,
        max_area_factor=max_area_factor, rtol=rtol)

    model = _start_model(model_ref, area_m2_start, ys,
                         adjust_term_elev=adjust_term_elev)

    diag_path = gdir.get_filepath('model_diagnostics',
                                  filesuffix=output_filesuffix,
                                  delete=True)
    vas_diag_path = gdir.get_filepath('vas_diagnostics',
                                      filesuffix=output_filesuffix,
                                      delete=True)
    model.run_until_and_store(
        year_end=ye, diag_path=diag_path, vas_diag_path=vas_diag_path,
        instant_geometry_change=instant_geometry_change)

    gdir.add_to_diagnostics('vas_reconstruction_start_yr', int(ys))
    gdir.add_to_diagnostics('vas_reconstruction_start_area_m2',
                            float(area_m2_start))
    gdir.add_to_diagnostics('vas_reconstruction_target_yr', int(target_yr))

    return model


def _dmdtda_from_reconstruction(gdir, mb_model, yr0, yr1, ys, target_yr,
                                adjust_term_elev=False,
                                instant_geometry_change=False,
                                max_area_factor=100, rtol=1e-4):
    """Geodetic mass balance of a reconstructed glacier, in kg m-2 yr-1.

    Same convention as OGGM's dynamic melt_f calibration: the mass change
    between `yr0` and `yr1` divided by the (fixed) RGI area and by the length
    of the period.
    """
    model_ref = _reference_model(gdir, mb_model, target_yr)
    area_m2_start = find_start_area_from_model(
        model_ref, ys, target_yr=target_yr,
        adjust_term_elev=adjust_term_elev,
        instant_geometry_change=instant_geometry_change,
        max_area_factor=max_area_factor, rtol=rtol)
    model = _start_model(model_ref, area_m2_start, ys,
                         adjust_term_elev=adjust_term_elev)

    model.run_until(yr0, instant_geometry_change=instant_geometry_change)
    volume_yr0 = model.volume_m3
    model.run_until(yr1, instant_geometry_change=instant_geometry_change)
    volume_yr1 = model.volume_m3

    dmdtda = ((volume_yr1 - volume_yr0) * model.rho / gdir.rgi_area_m2 /
              (yr1 - yr0))
    return dmdtda, area_m2_start


@entity_task(log, writes=['mb_calib'])
def mb_calibration_dynamic_from_geodetic_mb(gdir, *,
                                            ref_mb=None, ref_mb_err=None,
                                            ref_mb_period=None,
                                            ys=None, target_yr=None,
                                            melt_f_min=None, melt_f_max=None,
                                            prcp_fac=None, temp_bias=None,
                                            adjust_term_elev=False,
                                            instant_geometry_change=False,
                                            max_area_factor=100, rtol=1e-4,
                                            write_to_gdir=True,
                                            overwrite_gdir=False,
                                            **kwargs):
    """Calibrate `melt_f` on geodetic MB using the *evolving* glacier.

    :py:func:`mb_calibration_from_geodetic_mb` matches the observed mass
    change with the mass balance evaluated on the fixed RGI geometry. That
    ignores two things: the glacier changes shape during the observation
    period, and the RGI date is not the start of that period - it ranges from
    the 1960s to the 2010s across the inventory, so for most glaciers the RGI
    geometry is not the geometry the satellites saw.

    This task removes both approximations. For each trial `melt_f` it
    reconstructs the glacier from `ys`, constrained to reproduce the RGI area
    at the inventory date, and compares the modelled mass change over the
    reference period against the observation. It is OGGM's dynamic melt_f
    calibration, but the "spinup" is a one dimensional root find on the start
    area rather than a flowline run, so it costs milliseconds.

    Only `melt_f` is calibrated. `prcp_fac` and `temp_bias` are taken as
    given, exactly as in OGGM's dynamic calibration.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
    ref_mb : float, optional
        the reference mass balance to match [kg m-2 yr-1]. Defaults to
        Hugonnet et al. (2021).
    ref_mb_err : float, optional
        its error [kg m-2 yr-1], stored but not used in the calibration
    ref_mb_period : str, optional
        e.g. '2000-01-01_2020-01-01'. Defaults to
        `settings['geodetic_mb_period']`.
    ys : int, optional
        the year to reconstruct from. Defaults to the first year of the
        climate record.
    target_yr : int, optional
        the year at which to match the RGI area. Defaults to the year after
        the inventory date.
    melt_f_min, melt_f_max : float, optional
        bounds for the melt factor. Default to the settings.
    prcp_fac : float, optional
        the precipitation factor to use. Defaults to the settings, or to
        `decide_winter_precip_factor` if those leave it open.
    temp_bias : float, optional
        the temperature bias to use. Defaults to the settings, else 0.
    adjust_term_elev : bool, optional
        whether to move the terminus elevation with the start area
    instant_geometry_change : bool, optional
        whether to neglect the response time scales
    max_area_factor : float, optional
        the largest start area to consider, as a multiple of the RGI area
    rtol : float, optional
        the relative area error that counts as a geometry match
    write_to_gdir : bool, optional
        whether to write the calibrated parameters to the glacier settings
    overwrite_gdir : bool, optional
        whether to overwrite an existing calibration
    **kwargs
        passed to :py:class:`VAScalingMassBalance`

    Returns
    -------
    dict
        the calibrated parameters

    """
    if melt_f_min is None:
        melt_f_min = gdir.settings['melt_f_min']
    if melt_f_max is None:
        melt_f_max = gdir.settings['melt_f_max']

    # the reference mass balance
    if ref_mb_period is None:
        ref_mb_period = gdir.settings['geodetic_mb_period']
    if ref_mb is None:
        df = utils.get_geodetic_mb_dataframe(
            rgi_version=gdir.rgi_version).loc[gdir.rgi_id]
        df = df.loc[df['period'] == ref_mb_period]
        if len(df) == 0:
            raise InvalidWorkflowError(
                f'{gdir.rgi_id}: no geodetic mass balance available for the '
                f'period {ref_mb_period}.')
        # dmdtda is in m w.e. yr-1
        ref_mb = float(df['dmdtda'].iloc[0]) * 1000
        if ref_mb_err is None:
            ref_mb_err = float(df['err_dmdtda'].iloc[0]) * 1000

    yr0, yr1 = [int(y.split('-')[0]) for y in ref_mb_period.split('_')]

    # the parameters we are not calibrating
    if prcp_fac is None:
        if gdir.settings['prcp_fac'] is None:
            prcp_fac = massbalance.decide_winter_precip_factor(gdir)
        else:
            prcp_fac = gdir.settings['prcp_fac']
    if temp_bias is None:
        try:
            temp_bias = gdir.settings['temp_bias']
        except KeyError:
            temp_bias = 0

    def _mb_model(melt_f):
        return VAScalingMassBalance(gdir, melt_f=melt_f, prcp_fac=prcp_fac,
                                    temp_bias=temp_bias,
                                    check_calib_params=False, **kwargs)

    mbmod = _mb_model(melt_f_min)
    if ys is None:
        ys = int(mbmod.ys)
    target_yr = _target_year(gdir, mbmod, target_yr)
    if ys > yr0:
        raise InvalidParamsError(
            f'{gdir.rgi_id}: the reconstruction has to start before the '
            f'reference period, but ys={ys} is after {yr0}.')
    if ys >= target_yr:
        raise InvalidParamsError(
            f'{gdir.rgi_id}: the reconstruction has to start before the date '
            f'the geometry is matched at, but ys={ys} is not before '
            f'{target_yr}.')
    if not (mbmod.is_year_valid(yr0) and mbmod.is_year_valid(yr1 - 1)):
        raise InvalidWorkflowError(
            f'{gdir.rgi_id}: the reference period {ref_mb_period} is not '
            f'covered by the climate data [{mbmod.ys}, {mbmod.ye}].')

    state = {}

    def _cost(melt_f):
        dmdtda, area_m2_start = _dmdtda_from_reconstruction(
            gdir, _mb_model(melt_f), yr0, yr1, ys, target_yr,
            adjust_term_elev=adjust_term_elev,
            instant_geometry_change=instant_geometry_change,
            max_area_factor=max_area_factor, rtol=rtol)
        state['area_m2_start'] = area_m2_start
        state['dmdtda'] = dmdtda
        return dmdtda - ref_mb

    def _feasible(melt_f):
        try:
            _cost(melt_f)
            return True
        except RuntimeError:
            return False

    # a large melt_f can ask for a start glacier bigger than we allow. Shrink
    # the upper bound to the largest melt_f we can actually reconstruct.
    if not _feasible(melt_f_max):
        if not _feasible(melt_f_min):
            raise RuntimeError(
                f'{gdir.rgi_id}: no melt factor in '
                f'[{melt_f_min}, {melt_f_max}] allows a reconstruction from '
                f'{ys}. Try a later `ys` or a larger `max_area_factor`.')
        lo, hi = melt_f_min, melt_f_max
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if _feasible(mid):
                lo = mid
            else:
                hi = mid
        log.warning('(%s) melt_f_max reduced from %.2f to %.2f: a larger '
                    'melt factor cannot be reconstructed from %d.',
                    gdir.rgi_id, melt_f_max, lo, ys)
        melt_f_max = lo

    try:
        melt_f = brentq(_cost, melt_f_min, melt_f_max, xtol=1e-4)
        mismatch = _cost(melt_f)
    except ValueError:
        # no sign change in the bracket: pin to the closer bound, as
        # `mb_calibration_from_scalar_mb` does
        cost_min = _cost(melt_f_min)
        cost_max = _cost(melt_f_max)
        melt_f = melt_f_min if abs(cost_min) < abs(cost_max) else melt_f_max
        mismatch = _cost(melt_f)
        log.warning('(%s) could not match the geodetic mass balance with a '
                    'melt factor in [%.2f, %.2f]: using %.2f, which is off '
                    'by %.1f kg m-2 yr-1.', gdir.rgi_id, melt_f_min,
                    melt_f_max, melt_f, mismatch)

    df = dict()
    df['rgi_id'] = gdir.rgi_id
    df['bias'] = 0
    df['melt_f'] = melt_f
    df['prcp_fac'] = prcp_fac
    df['temp_bias'] = temp_bias
    df['reference_mb'] = ref_mb
    df['reference_mb_err'] = ref_mb_err
    df['reference_period'] = ref_mb_period
    df['mb_global_params'] = massbalance._mb_global_params_from_model(
        _mb_model(melt_f))
    df['baseline_climate_source'] = gdir.get_climate_info(
        filename=mbmod.filename,
        input_filesuffix=mbmod.input_filesuffix)['baseline_climate_source']
    # what the dynamic calibration did on top of the static one
    df['vas_dynamic_calibration'] = True
    df['vas_reconstruction_start_yr'] = int(ys)
    df['vas_reconstruction_start_area_m2'] = float(state['area_m2_start'])
    df['vas_reconstruction_target_yr'] = int(target_yr)
    df['vas_dmdtda_mismatch'] = float(mismatch)

    if write_to_gdir:
        stored = gdir.get_stored_settings()
        if any(k in stored for k in ['melt_f', 'prcp_fac', 'temp_bias']) \
                and not overwrite_gdir:
            raise InvalidWorkflowError(
                'There are already mass balance parameters stored in the '
                'settings file. Set `overwrite_gdir` to True if you want to '
                'overwrite a previous calibration.')
        for key, value in df.items():
            gdir.settings[key] = value

    return df


class _VASMassBalanceWrapper(MassBalanceModel):
    """Common logic for the constant and random VAS mass balance models.

    Both wrap a :py:class:`VAScalingMassBalance` and only change *which*
    climate year is used. Everything else - the parameters, the geometry
    (`min_hgt` / `max_hgt`) and the turnover (`prcp_clim`) - is proxied to
    the wrapped model, so that `VAScalingModel` can drive either of them.
    """

    def __init__(self, gdir, **kwargs):
        super(_VASMassBalanceWrapper, self).__init__(gdir=gdir)
        self.mbmod = VAScalingMassBalance(gdir, **kwargs)
        self.hemisphere = gdir.hemisphere
        self.valid_bounds = None

    # -- geometry, driven by VAScalingModel during a run

    @property
    def min_hgt(self):
        """Glacier terminus elevation [m asl.]"""
        return self.mbmod.min_hgt

    @min_hgt.setter
    def min_hgt(self, value):
        self.mbmod.min_hgt = value

    @property
    def min_hgt_0(self):
        """Glacier terminus elevation at the RGI date [m asl.]"""
        return self.mbmod.min_hgt_0

    @property
    def max_hgt(self):
        """Maximum glacier surface elevation [m asl.]"""
        return self.mbmod.max_hgt

    @max_hgt.setter
    def max_hgt(self, value):
        self.mbmod.max_hgt = value

    @property
    def prcp_clim(self):
        """Climatological solid precipitation [m w.e. yr-1]"""
        return self.mbmod.prcp_clim

    # -- mass balance parameters

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        self.mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        self.mbmod.prcp_fac = value

    @property
    def melt_f(self):
        """Melt factor [kg m-2 day-1 K-1]"""
        return self.mbmod.melt_f

    @melt_f.setter
    def melt_f(self, value):
        self.mbmod.melt_f = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.bias

    @bias.setter
    def bias(self, value):
        self.mbmod.bias = value

    def is_year_valid(self, year):
        """Any year can be simulated with a constant or random climate."""
        return True

    def get_ela(self, year=None, **kwargs):
        """The ELA cannot be computed by this mass balance model."""
        raise NotImplementedError('The equilibrium line altitude can not be '
                                  'computed for the volume/area scaling '
                                  'mass balance models.')


class RandomVASMassBalance(_VASMassBalanceWrapper):
    """Random shuffle of all MB years within a given time period.

    This is useful for finding a possible past glacier state or for
    sensitivity experiments.

    Note that this is going to be sensitive to extreme years in certain
    periods, but it is by far more physically reasonable than other
    approaches based on gaussian assumptions.
    """

    def __init__(self, gdir, y0=None, halfsize=15, seed=None,
                 all_years=False, unique_samples=False, **kwargs):
        """Initialize.

        Parameters
        ----------
        gdir : :py:class:`oggm.GlacierDirectory`
        y0 : int, required unless `all_years` is set
            the year at the center of the period of interest
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        seed : int, optional
            random seed used to initialize the pseudo-random number generator
        all_years : bool
            if True, overrides `y0` and `halfsize` to use all available years
        unique_samples : bool
            if True, chosen random mass balance years will only be available
            once per random climate period-length. If False, every model year
            is chosen from the random climate period with the same probability
        **kwargs
            passed to :py:class:`VAScalingMassBalance`

        """
        super(RandomVASMassBalance, self).__init__(gdir, **kwargs)

        self.valid_bounds = None
        self.rng = np.random.RandomState(seed)
        self._state_yr = dict()

        if all_years:
            self.years = self.mbmod.years
        else:
            if y0 is None:
                raise InvalidParamsError('Please set `y0` explicitly')
            self.years = np.arange(y0 - halfsize, y0 + halfsize + 1)
        self.yr_range = (self.years[0], self.years[-1] + 1)
        self.ny = len(self.years)

        self.unique_samples = unique_samples
        self.sampling_years = self.years

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

    def get_monthly_mb(self, heights=None, year=None, **kwargs):
        """Glacier wide mass balance for the given month [m ice s-1]."""
        ryr, m = floatyear_to_date(year)
        ryr = date_to_floatyear(self.get_state_yr(ryr), m)
        return self.mbmod.get_monthly_mb(heights, year=ryr, **kwargs)

    def get_annual_mb(self, heights=None, year=None, **kwargs):
        """Glacier wide mass balance for the given year [m ice s-1]."""
        ryr = self.get_state_yr(int(year))
        return self.mbmod.get_annual_mb(heights, year=ryr, **kwargs)

    def get_specific_mb(self, heights=None, widths=None, fls=None, year=None,
                        time_resolution='annual'):
        """Glacier wide specific mass balance [mm w.e. yr-1]."""
        out = [self.mbmod.get_specific_mb(year=self.get_state_yr(int(yr)),
                                          time_resolution=time_resolution)
               for yr in np.atleast_1d(year)]
        return utils.set_array_type(out)


@entity_task(log)
def run_random_climate(gdir, nyears=1000, y0=None, halfsize=15,
                       bias=0, seed=None, temperature_bias=None,
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
        fp = gdir.get_filepath('vas_diagnostics',
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
    vas_diag_path = gdir.get_filepath('vas_diagnostics',
                                      filesuffix=output_filesuffix,
                                      delete=True)
    # run model
    model.run_until_and_store(year_end=nyears, diag_path=diag_path,
                              vas_diag_path=vas_diag_path, **kwargs)

    return model


class ConstantVASMassBalance(_VASMassBalanceWrapper):
    """Constant mass balance during a chosen period.

    This is useful for equilibrium experiments. Unlike OGGM's
    `ConstantMassBalance`, no interpolation over elevation bins is needed:
    the VAS model returns a single glacier wide value, so the average over
    the period is computed directly.
    """

    def __init__(self, gdir, y0=None, halfsize=15, **kwargs):
        """Initialize.

        Parameters
        ----------
        gdir : :py:class:`oggm.GlacierDirectory`
        y0 : int, required
            the year at the center of the period of interest
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        **kwargs
            passed to :py:class:`VAScalingMassBalance`

        """
        super(ConstantVASMassBalance, self).__init__(gdir, **kwargs)

        if y0 is None:
            raise InvalidParamsError('Please set `y0` explicitly')

        self.valid_bounds = None
        self.y0 = y0
        self.halfsize = halfsize
        self.years = np.arange(y0 - halfsize, y0 + halfsize + 1)
        self.ny = len(self.years)

    def get_monthly_mb(self, heights=None, year=None, **kwargs):
        """Average glacier wide mass balance for the given month
        over the climate period [m ice s-1]."""
        _, m = floatyear_to_date(year)
        out = [self.mbmod.get_monthly_mb(heights,
                                         year=date_to_floatyear(yr, m),
                                         **kwargs)
               for yr in self.years]
        return np.mean(out, axis=0)

    def get_annual_mb(self, heights=None, year=None, **kwargs):
        """Average annual glacier wide mass balance over the climate
        period [m ice s-1]."""
        out = [self.mbmod.get_annual_mb(heights, year=yr, **kwargs)
               for yr in self.years]
        return np.mean(out, axis=0)

    def get_specific_mb(self, heights=None, widths=None, fls=None, year=None,
                        time_resolution='annual'):
        """Average glacier wide specific mass balance over the climate
        period [mm w.e. yr-1]. Independent of `year`."""
        out = np.mean(self.mbmod.get_specific_mb(
            year=self.years, time_resolution=time_resolution))
        if year is None or np.size(year) == 1:
            return out
        return np.repeat(out, np.size(year))


@entity_task(log)
def run_constant_climate(gdir, nyears=1000, y0=None, halfsize=15,
                         bias=0, temperature_bias=None,
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
        fp = gdir.get_filepath('vas_diagnostics',
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
    mb_mod = ConstantVASMassBalance(gdir, bias=bias, y0=y0,
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
    vas_diag_path = gdir.get_filepath('vas_diagnostics',
                                      filesuffix=output_filesuffix,
                                      delete=True)
    # run model
    model.run_until_and_store(year_end=nyears, diag_path=diag_path,
                              vas_diag_path=vas_diag_path, **kwargs)

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

        # ice density: keep it consistent with the mass balance model
        self.rho = getattr(mb_model, 'ice_density', None)
        if self.rho is None:
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
            raise InvalidParamsError("Glacier type can only be 'Glacier' "
                                     "or 'Ice cap'.")

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
        self.mb_model.max_hgt = self.max_hgt
        # year_0 can legitimately sit outside of the climate record (e.g.
        # `find_start_area` compares at the RGI date), and the initial
        # specific mass balance is only a diagnostic - the first step
        # recomputes it anyway.
        if self.mb_model.is_year_valid(self.year):
            self._get_specific_mb()
        else:
            self.spec_mb = np.nan
        # create geometry change parameters
        self.dL = 0
        self.dA = 0
        self.dV = 0

        # create time scale parameters
        self.tau_a = 1
        self.tau_l = 1

    def _get_specific_mb(self):
        """Invoke `get_specific_mb()` from the mass balance model for the
        current year and glacier terminus elevation.

        The VAS mass balance models carry the glacier geometry themselves, so
        the current terminus elevation is handed over before asking for the
        mass balance.
        """
        self.mb_model.min_hgt = self.min_hgt
        self.spec_mb = self.mb_model.get_specific_mb(year=self.year)

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
                            vas_diag_path=None,
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
        diag_path : str, optional
            path where to store the OGGM compatible glacier diagnostics
            (volume, area and length), default = None
        vas_diag_path : str, optional
            path where to store the full VAS diagnostics, which additionally
            hold the specific mass balance, the terminus and maximum surface
            elevation and the response time scales, default = None
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

        # the model is annual, and OGGM is calendar year based
        years = np.arange(int(np.floor(self.year)),
                          int(np.floor(year_end)) + 1)
        ny = len(years)

        # initialize diagnostics output file
        diag_ds = xr.Dataset()

        # Global attributes
        diag_ds.attrs['description'] = 'VAS model output'
        diag_ds.attrs['oggm_version'] = __version__
        diag_ds.attrs['calendar'] = '365-day no leap'
        diag_ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S",
                                                  gmtime())
        diag_ds.attrs['hemisphere'] = self.mb_model.hemisphere
        diag_ds.attrs['glacier_type'] = self.glacier_type

        # Coordinates - same convention as OGGM, so that the usual tools
        # (`utils.compile_run_output`) can read our output
        yrs, months = utils.floatyear_to_date(years.astype(np.float64))
        sm = cfg.PARAMS['hydro_month_' + self.mb_model.hemisphere]
        hyrs, hmonths = utils.calendardate_to_hydrodate(yrs, months,
                                                        start_month=sm)

        diag_ds.coords['time'] = ('time', years.astype(np.float64))
        diag_ds.coords['calendar_year'] = ('time', yrs)
        diag_ds.coords['calendar_month'] = ('time', months)
        diag_ds.coords['hydro_year'] = ('time', hyrs)
        diag_ds.coords['hydro_month'] = ('time', hmonths)
        # add description as attribute to coordinates
        diag_ds['time'].attrs['description'] = 'Floating year'
        diag_ds['calendar_year'].attrs['description'] = 'Calendar year'
        diag_ds['calendar_month'].attrs['description'] = 'Calendar month'
        diag_ds['hydro_year'].attrs['description'] = 'Hydrological year'
        diag_ds['hydro_month'].attrs['description'] = 'Hydrological month'

        # create empty variables and attributes
        diag_ds['volume_m3'] = ('time', np.zeros(ny) * np.nan)
        diag_ds['volume_m3'].attrs['description'] = 'Total glacier volume'
        diag_ds['volume_m3'].attrs['unit'] = 'm 3'
        diag_ds['area_m2'] = ('time', np.zeros(ny) * np.nan)
        diag_ds['area_m2'].attrs['description'] = 'Total glacier area'
        diag_ds['area_m2'].attrs['unit'] = 'm 2'
        diag_ds['length_m'] = ('time', np.zeros(ny) * np.nan)
        diag_ds['length_m'].attrs['description'] = 'Glacier length'
        diag_ds['length_m'].attrs['unit'] = 'm'
        diag_ds['spec_mb'] = ('time', np.zeros(ny) * np.nan)
        diag_ds['spec_mb'].attrs['description'] = 'Specific mass balance'
        diag_ds['spec_mb'].attrs['unit'] = 'mm w.e. yr-1'
        diag_ds['min_hgt'] = ('time', np.zeros(ny) * np.nan)
        diag_ds['min_hgt'].attrs['description'] = 'Terminus elevation'
        diag_ds['min_hgt'].attrs['unit'] = 'm asl.'
        diag_ds['max_hgt'] = ('time', np.zeros(ny) * np.nan)
        diag_ds['max_hgt'].attrs['description'] = 'Maximum surface elevation'
        diag_ds['max_hgt'].attrs['unit'] = 'm asl.'
        diag_ds['tau_l'] = ('time', np.zeros(ny) * np.nan)
        diag_ds['tau_l'].attrs['description'] = 'Length change response time'
        diag_ds['tau_l'].attrs['unit'] = 'years'
        diag_ds['tau_a'] = ('time', np.zeros(ny) * np.nan)
        diag_ds['tau_a'].attrs['description'] = 'Area change response time'
        diag_ds['tau_a'].attrs['unit'] = 'years'

        # run the model
        for i, yr in enumerate(years):
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

        if vas_diag_path is not None:
            # the full VAS output, including the response times
            diag_ds.to_netcdf(vas_diag_path)

        if diag_path is not None:
            # `utils.compile_run_output` refuses variables it does not know
            # about, so the OGGM-compatible file only holds the geometry
            oggm_ds = diag_ds[OGGM_DIAG_VARS]
            # it also reads these ice dynamics parameters unconditionally.
            # They have no meaning for a volume/area scaling model.
            for attr in ['water_level', 'glen_a', 'fs']:
                oggm_ds.attrs[attr] = np.nan
            oggm_ds.to_netcdf(diag_path)

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
                      self.max_hgt, self.mb_model,
                      glacier_type=self.glacier_type)

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
        self.year_0 = self.year
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
        # adjust date according to the floating year convention
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
