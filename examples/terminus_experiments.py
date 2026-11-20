"""Terminus experiments: three ways to place the terminus, three glaciers.

Runs the three Alpine test glaciers (elevation band prepro directories,
GSWP3-W5E5, OGGM's informed three-step calibration on Hugonnet et al. 2021)
and writes everything to one json file in the working directory. The three
terminus options are:

- ``eq8_start``: Eq. (8) of Marzeion et al. (2012) anchored on the glacier at
  the start of the run (Ben's MATLAB script and the previous python port);
- ``eq8_rgi``: Eq. (8) anchored on the RGI glacier (the paper's text),
  unbounded below the tongue apart from a floor far below;
- ``terrain``: the terminus follows the flowline and downstream line
  (``TerminusProfile``), anchored on the RGI glacier.

For each glacier and option the file holds:

1. terminus elevation against glacier length;
2. the reconstruction from 1901 (plus, for the terrain option, a run where
   the anchor is iterated so the run also hits the observed terminus);
3. the area at the RGI date as a function of the 1901 area, and the 1901
   areas that reproduce the RGI area;
4. area trajectories from a few 1901 areas;
5. the modelled 2000-2019 mass balance and the response times at the RGI
   date of each reconstruction.

See "Open problem 3" in the README. Run with
``python examples/terminus_experiments.py``; it downloads the same prepro
directories as ``run_alps.py``.
"""
import json
import os
import warnings

import numpy as np
from scipy.optimize import brentq

from oggm import cfg, utils, workflow
import oggm_vas as vas
from oggm_vas import core as c

warnings.simplefilter('ignore')
vas.initialize(logging_level='CRITICAL')
WD = utils.mkdir(os.path.join(os.path.expanduser('~'), 'OGGM',
                              'vas_terminus'))
cfg.PATHS['working_dir'] = WD
BASE_URL = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/'
            'L3-L5_files/2023.3/elev_bands/W5E5/')
TEMP_BIAS_URL = ('https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/'
                 'oggm_v1.6/w5e5_rgi6_perglacier_temp_bias_v2025.6.2.csv')
RGI_IDS = ['RGI60-11.00897', 'RGI60-11.00787', 'RGI60-11.00746']
YS, YE = 1901, 2020
OPTIONS = ['eq8_start', 'eq8_rgi', 'terrain']


def series(ds):
    return {'area': [round(float(x) / 1e6, 4) for x in ds.area_m2.values],
            'term': [round(float(x), 1) for x in ds.min_hgt.values]}


def summary(ds, target):
    # spec_mb at time t is the mass balance of year t-1
    mb = float(ds.spec_mb.sel(time=slice(2001, 2020)).mean())
    r = ds.sel(time=target)
    return {'mb_2000_2019': round(mb), 'tau_l': round(float(r.tau_l), 1),
            'tau_a': round(float(r.tau_a), 1),
            'term_target': round(float(r.min_hgt)),
            'area_1901': round(float(ds.area_m2.sel(time=YS)) / 1e6, 3)}


def straight_line(zmin, zmax, L0):
    """Eq. (8) anchored on the RGI glacier, as a TerminusProfile."""
    dx = 20.
    d = (np.arange(int(6 * L0 / dx)) + 0.5) * dx
    return vas.TerminusProfile(zmax + (zmin - zmax) * d / L0, dx, L0, L0,
                               zmin, max_hgt=zmax)


def area_at_target(ref, a_m2, target):
    m = c._start_model(ref, a_m2, YS)
    m.run_until(target)
    return m.area_m2


out = {'years': list(range(YS, YE + 1)), 'glaciers': []}
gdirs = workflow.init_glacier_directories(RGI_IDS, from_prepro_level=3,
                                          prepro_border=160,
                                          prepro_base_url=BASE_URL)
workflow.execute_entity_task(vas.terminus_profile_from_flowlines, gdirs)
for gd in gdirs:
    vas.mb_calibration_from_geodetic_mb(gd, overwrite_gdir=True,
                                        informed_threestep=True,
                                        temp_bias_file_path=TEMP_BIAS_URL)
    mbmod = vas.VAScalingMassBalance(gd)
    target = c._target_year(gd, mbmod, None)
    zmin, zmax = vas.get_min_max_elevation(gd)
    tp = vas.TerminusProfile.from_gdir(gd)
    L0 = tp.length_m_ref

    refs = {}
    for name in OPTIONS:
        ref = c._reference_model(gd, mbmod, target)
        ref.terminus_profile = {'eq8_start': None,
                                'eq8_rgi': straight_line(zmin, zmax, L0),
                                'terrain': tp}[name]
        refs[name] = ref

    g = {'rgi_id': gd.rgi_id, 'name': gd.name, 'rgi_area': gd.rgi_area_km2,
         'rgi_year': c._rgi_year(gd), 'target_yr': target,
         'zmin': float(zmin), 'zmax': float(zmax), 'L0': L0 / 1e3,
         'flowline_length': tp.glacier_length_m / 1e3,
         'ref_mb': float(gd.observations['ref_mb']['value']),
         'ref_mb_err': float(gd.observations['ref_mb']['err']),
         'params': {k: round(float(gd.settings[k]), 3)
                    for k in ['melt_f', 'prcp_fac', 'temp_bias']}}
    print(gd.rgi_id, gd.name, g['params'])

    # 1. terminus vs length
    Ls = np.linspace(0, 2.5 * L0, 121)
    g['profile'] = {'length': [round(x / 1e3, 4) for x in Ls],
                    'terrain': [round(tp(x), 1) for x in Ls],
                    'eq8_rgi': [round(refs['eq8_rgi'].terminus_profile(x), 1)
                                for x in Ls]}

    # 2. reconstructions and 5. their summaries
    rec, summ = {}, {}
    for name in OPTIONS:
        try:
            A = c.find_start_area_from_model(refs[name], YS,
                                             target_yr=target)
            ds = c._start_model(refs[name], A, YS).run_until_and_store(YE)
            rec[name], summ[name] = series(ds), summary(ds, target)
        except Exception as e:
            print('  rec', name, 'failed:', str(e)[:100])
            rec[name] = summ[name] = None
    # terrain, with the anchor (the length that sits on the observed
    # terminus) moved until the run also hits the observed terminus at the
    # RGI date. The terminus error jumps where the start-area search changes
    # branch, so bisect on its sign and accept the result only if the error
    # is small.
    def run_anchor(fac):
        tp.length_m_ref = fac * L0
        try:
            A = c.find_start_area_from_model(refs['terrain'], YS,
                                             target_yr=target)
        except Exception:
            return None, np.nan
        ds = c._start_model(refs['terrain'], A, YS).run_until_and_store(YE)
        return ds, float(ds.min_hgt.sel(time=target)) - zmin

    rec['terrain_matched'] = summ['terrain_matched'] = None
    facs = np.arange(0.8, 1.41, 0.05)
    errs = [run_anchor(f)[1] for f in facs]
    best = None
    for i in range(len(facs) - 1):
        if np.sign(errs[i]) != np.sign(errs[i + 1]):
            lo, hi, elo = facs[i], facs[i + 1], errs[i]
            for _ in range(30):
                mid = (lo + hi) / 2
                e = run_anchor(mid)[1]
                if np.sign(e) == np.sign(elo):
                    lo, elo = mid, e
                else:
                    hi = mid
            for f in (lo, hi):
                ds, e = run_anchor(f)
                if ds is not None and (best is None or abs(e) < abs(best[1])):
                    best = (ds, e, f)
            break
    tp.length_m_ref = L0
    if best is not None and abs(best[1]) < 10:
        rec['terrain_matched'] = series(best[0])
        summ['terrain_matched'] = summary(best[0], target)
        summ['terrain_matched']['anchor_factor'] = round(best[2], 4)
    print('  terrain matched:', None if best is None else
          f'error {best[1]:+.1f} m at {best[2]:.4f} L0',
          'accepted' if rec['terrain_matched'] else 'rejected')
    g['rec'], g['summary'] = rec, summ

    # 3. area at the RGI date against the 1901 area, and all the 1901 areas
    # that reproduce the RGI area
    starts = np.geomspace(0.1, 80, 61)
    scan = {'start': [round(float(a), 4) for a in starts]}
    roots = {}
    for name in OPTIONS:
        vals = np.array([area_at_target(refs[name], a * 1e6, target)
                         for a in starts]) / 1e6
        scan[name] = [round(float(v), 4) for v in vals]
        err = vals - gd.rgi_area_km2
        roots[name] = []
        for i in np.where(np.sign(err[:-1]) != np.sign(err[1:]))[0]:
            f = lambda a: area_at_target(refs[name], a * 1e6, target) / 1e6 \
                - gd.rgi_area_km2
            roots[name].append(round(brentq(f, starts[i], starts[i + 1],
                                            xtol=1e-3), 3))
    g['scan'], g['roots'] = scan, roots
    print('  roots:', roots)

    # 4. trajectories from a few start areas
    traj = {'start_factor': [0.25, 0.5, 1, 2, 4]}
    for name in OPTIONS:
        traj[name] = []
        for f in traj['start_factor']:
            ds = c._start_model(refs[name], f * gd.rgi_area_m2,
                                YS).run_until_and_store(YE)
            traj[name].append([round(float(x) / 1e6, 4)
                               for x in ds.area_m2.values])
    g['traj'] = traj

    out['glaciers'].append(g)

fp = os.path.join(WD, 'terminus_experiments.json')
with open(fp, 'w') as f:
    json.dump(out, f)
print('written to', fp)
