"""Transliteration of the time stepping in matlab-code/model_cru_hugonnet.m,
with a switch for each place where it differs from oggm_vas.core.

1. With all switches on the "python" side it must reproduce VAScalingModel.
2. Then each switch is flipped to the MATLAB behaviour, one at a time, and
   the 1901 start area of Hintereisferner is recomputed.

Offline: uses the HISTALP demo data of the test suite (1802-2002) and the
same made up reference MB (-500 kg m-2 yr-1 over 1953-2002, prcp_fac 2.5).
"""
import os
import shutil

import numpy as np
import geopandas as gpd
from scipy.optimize import brentq

import oggm
import oggm.cfg as cfg
from oggm.utils import get_demo_file, date_to_floatyear
from oggm.core import gis, climate, centerlines, massbalance
from oggm.tests.funcs import get_test_dir
import oggm_vas as vascaling

REF_MB = -500.
REF_YEARS = (1953, 2003)
YS = 1901

testdir = os.path.join(get_test_dir(), 'tmp_vas_matlab')
shutil.rmtree(testdir, ignore_errors=True)
os.makedirs(testdir)

vascaling.initialize(logging_level='ERROR')
cfg.PATHS['working_dir'] = testdir
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['use_intersects'] = False
cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
cfg.PARAMS['baseline_climate'] = 'CUSTOM'
cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
cfg.PARAMS['prcp_fac'] = 2.5
cfg.PARAMS['use_multiple_flowlines'] = False
cfg.PARAMS['border'] = 40

entity = gpd.read_file(get_demo_file('Hintereisferner_RGI6.shp')).iloc[0]
gdir = oggm.GlacierDirectory(entity, base_dir=testdir)
gis.define_glacier_region(gdir)
gis.glacier_masks(gdir)
centerlines.compute_centerlines(gdir)
centerlines.initialize_flowlines(gdir)
centerlines.catchment_area(gdir)
centerlines.catchment_intersections(gdir)
centerlines.catchment_width_geom(gdir)
centerlines.catchment_width_correction(gdir)
climate.process_custom_climate_data(gdir)
massbalance.mb_calibration_from_scalar_mb(
    gdir, ref_mb=REF_MB, ref_mb_years=REF_YEARS,
    mb_model_class=vascaling.VAScalingMassBalance,
    calibrate_param1='melt_f', calibrate_param2='prcp_fac',
    calibrate_param3='temp_bias', overwrite_gdir=True,
    overwrite_observations=True)

mbmod = vascaling.VAScalingMassBalance(gdir)
zmin, zmax = vascaling.get_min_max_elevation(gdir)
target_yr = vascaling.core._target_year(gdir, mbmod)
print('melt_f %.4f prcp_fac %.3f temp_bias %.3f | zmin %.0f zmax %.0f | '
      'rgi area %.3f km2 | target year %d | climate %d-%d'
      % (mbmod.melt_f, mbmod.prcp_fac, mbmod.temp_bias, zmin, zmax,
         gdir.rgi_area_km2, target_yr, mbmod.ys, mbmod.ye))

# monthly climate at the reference height, biases already applied
years = np.arange(mbmod.ys, mbmod.ye + 1)
T, P, G, D = {}, {}, {}, {}
for y in years:
    pok = np.where(mbmod.years == y)[0]
    assert len(pok) == 12
    T[y] = np.asarray(mbmod.temp[pok], dtype=float)
    P[y] = np.asarray(mbmod.prcp[pok], dtype=float)
    G[y] = np.asarray(mbmod.grad[pok], dtype=float)
    D[y] = np.array([mbmod.days_in_month(date_to_floatyear(y, m))
                     for m in range(1, 13)], dtype=float)
ref_hgt = mbmod.ref_hgt
t_melt = mbmod.temp_melt
t_solid = mbmod.temp_all_solid

ca = cfg.PARAMS['vas_c_area_m2']
gamma = cfg.PARAMS['vas_gamma_area']
cl = cfg.PARAMS['vas_c_length_m']
q = cfg.PARAMS['vas_q_length']


def solid_prcp(y, zterm):
    t_term = T[y] + G[y] * (zterm - ref_hgt)
    f = 1 + (t_term - t_solid) / (G[y] * (zmax - zterm))
    return P[y] * np.clip(f, 0, 1)


def melt_sum(y, zterm, day_weight):
    t_term = T[y] + G[y] * (zterm - ref_hgt)
    w = D[y] if day_weight else 1.
    return np.sum(np.clip(t_term - t_melt, 0, None) * w)


def calibrate_mu(day_weight):
    """mu_rgi in the MATLAB script: closed form on the RGI geometry."""
    yrs = np.arange(*REF_YEARS)
    psol = sum(solid_prcp(y, zmin).sum() for y in yrs)
    pdd = sum(melt_sum(y, zmin, day_weight) for y in yrs)
    return (psol - len(yrs) * REF_MB) / pdd


def turnover(yrs):
    """Mean annual solid precipitation on the RGI geometry [m w.e. yr-1]."""
    return max(10e-3, np.mean([solid_prcp(y, zmin).sum() for y in yrs]) * 1e-3)


def run(area_start, y0, y1, solid_fixed=False, tau_a_from_v=False, rho=900.,
        sea_clamp=False, zero_when_v0=False, day_weight=True,
        turnover_ref_period=False, full=False):
    mu = calibrate_mu(day_weight)
    tov = turnover(np.arange(*REF_YEARS) if turnover_ref_period else years)
    A = area_start
    V = ca * A ** gamma
    L = (V / cl) ** (1 / q)
    L0 = L
    out = [(y0, A, V, L, zmin)]
    for y in range(y0, y1):
        # terminus: RGI elevation at the start of the run, then Eq. (8)
        zterm = zmax + L / L0 * (zmin - zmax)
        if sea_clamp:
            zterm = max(zterm, 0.)
        psol = solid_prcp(y, zmin if solid_fixed else zterm).sum()
        mb = psol - mu * melt_sum(y, zterm, day_weight)
        # time scales from the state at the start of the year
        if V == 0 or A == 0:
            tau_l = tau_a = 1.
        else:
            tau_l = max(1., V / (tov * A))
            if tau_a_from_v:
                a_over_l2 = (cl ** (2 / q) / ca ** (1 / gamma)
                             * V ** (1 / gamma - 2 / q))
            else:
                a_over_l2 = A / L ** 2
            tau_a = max(1., tau_l * a_over_l2)
        V = max(0., V + mb * A / rho)
        A = max(0., A + ((V / ca) ** (1 / gamma) - A) / tau_a)
        L = max(0., L + ((V / cl) ** (1 / q) - L) / tau_l)
        if zero_when_v0 and V == 0:
            A = L = 0.
        out.append((y + 1, A, V, L, zmax + L / L0 * (zmin - zmax)))
    out = np.array(out)
    return out if full else A


def start_area(**kw):
    def f(a):
        return 1 - run(a, YS, target_yr, **kw) / gdir.rgi_area_m2
    hi = gdir.rgi_area_m2
    while f(hi) > 0:
        hi *= 2
        if hi > 200 * gdir.rgi_area_m2:
            return np.nan
    return brentq(f, 100., hi, xtol=1.)


# ---- 1. the emulator with python switches against VAScalingModel
model_ref = vascaling.core._reference_model(gdir, mbmod, target_yr)
a_py = vascaling.find_start_area_from_model(model_ref, YS, target_yr=target_yr)
model = vascaling.core._start_model(model_ref, a_py, YS)
ref = [(model.year, model.area_m2, model.volume_m3, model.length_m,
        model.min_hgt)]
while model.year < target_yr:
    model.step()
    ref.append((model.year, model.area_m2, model.volume_m3, model.length_m,
                model.min_hgt))
ref = np.array(ref)
emu = run(a_py, YS, target_yr, full=True)
print('\nmu closed form (day weighted) %.6f vs OGGM melt_f %.6f'
      % (calibrate_mu(True), mbmod.melt_f))
print('turnover whole record %.6f vs prcp_clim %.6f'
      % (turnover(years), mbmod.prcp_clim))
for i, name in enumerate(['area', 'volume', 'length', 'min_hgt'], start=1):
    print('max rel diff %-8s %.2e'
          % (name, np.max(np.abs(emu[:, i] / ref[:, i] - 1))))
print('start area: VAScalingModel %.4f km2, emulator %.4f km2'
      % (a_py * 1e-6, start_area() * 1e-6))

# ---- 2. flip each switch to the MATLAB side
cases = [
    ('python (reference)', {}),
    ('solid precipitation fixed at RGI geometry', dict(solid_fixed=True)),
    ('tau_A from scaling A/L2 of V', dict(tau_a_from_v=True)),
    ('no ice density in dV', dict(rho=1000.)),
    ('terminus floor at sea level', dict(sea_clamp=True)),
    ('A = L = 0 as soon as V = 0', dict(zero_when_v0=True)),
    ('mu per month, not per day', dict(day_weight=False)),
    ('turnover over calibration period', dict(turnover_ref_period=True)),
    ('all of the above (= MATLAB)', dict(
        solid_fixed=True, tau_a_from_v=True, rho=1000., sea_clamp=True,
        zero_when_v0=True, day_weight=False, turnover_ref_period=True)),
]
print('\n%-44s %10s %10s %10s %9s' % ('', 'A_%d' % YS, 'A_1950', 'V_%d' % YS,
                                      'zterm_end'))
for name, kw in cases:
    a0 = start_area(**kw)
    o = run(a0, YS, target_yr, full=True, **kw)
    i50 = np.where(o[:, 0] == 1950)[0][0]
    print('%-44s %10.3f %10.3f %10.4f %9.0f'
          % (name, a0 * 1e-6, o[i50, 1] * 1e-6, o[0, 2] * 1e-9, o[-1, 4]))

shutil.rmtree(testdir, ignore_errors=True)
