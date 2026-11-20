"""Terminus experiments: Eq. (8) against the terrain-following terminus.

Runs the three Alpine test glaciers (centerline prepro directories, W5E5,
OGGM's informed three-step calibration) and writes everything to one json
file in the working directory:

1. terminus elevation against glacier length, for both terminus options;
2. the reconstruction from 1901 with each option, and with the terminus
   anchored so that the run hits the observed terminus at the RGI date;
3. the area at the RGI date as a function of the 1901 area;
4. area trajectories from a few 1901 areas;
5. the area at the RGI date when the 1901 glacier is in equilibrium with the
   1901-1930 climate plus a temperature bias dT.

See "Open problem 3" in the README. Run with python
examples/terminus_experiments.py; it downloads the same prepro directories
as run_alps.py.
"""
import warnings, json; warnings.simplefilter('ignore')
import numpy as np
from scipy.optimize import brentq
from oggm import cfg, workflow
import oggm_vas as vas
from oggm_vas import core as c
vas.initialize(logging_level='CRITICAL')
import os
from oggm import utils
WD = utils.mkdir(os.path.join(os.path.expanduser('~'), 'OGGM', 'vas_terminus'))
cfg.PATHS['working_dir'] = WD
BASE_URL = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/'
            'L3-L5_files/2023.3/centerlines/W5E5/')
URL = 'https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/oggm_v1.6/w5e5_rgi6_perglacier_temp_bias_v2025.6.2.csv'
IDS = {'RGI60-11.00897': 'Hintereisferner', 'RGI60-11.00787': 'RGI60-11.00787', 'RGI60-11.00746': 'RGI60-11.00746'}
YS, YE = 1901, 2020
SPIN = np.arange(1901, 1931)

def area_from_length(L):
    cl, ca, ql, gamma = c._scaling_params('Glacier')
    return (cl * L ** ql / ca) ** (1 / gamma)

def eq_area(mbmod, tp, dT):
    tb0 = mbmod.temp_bias
    mbmod.temp_bias = tb0 + dT
    def mb(z):
        mbmod.min_hgt = z
        return np.mean(mbmod.get_specific_mb(year=SPIN))
    try:
        zlo, zhi = tp.lowest_hgt, tp.max_hgt - 1
        if mb(zlo) > 0: z = zlo
        elif mb(zhi) < 0: return None
        else: z = brentq(mb, zlo, zhi, xtol=0.1)
    finally:
        mbmod.temp_bias = tb0
    Lmax = tp.length_m_ref + tp.distance_m[-1]
    L = brentq(lambda L: tp(L) - z - 1e-6, 1., Lmax) if tp(Lmax) < z else Lmax
    return area_from_length(L)

def series(ds):
    return {'area': [round(float(x) / 1e6, 4) for x in ds.area_m2.values],
            'term': [round(float(x), 1) for x in ds.min_hgt.values],
            'length': [round(float(x) / 1e3, 4) for x in ds.length_m.values]}

out = {'years': list(range(YS, YE + 1)), 'glaciers': []}
gdirs = workflow.init_glacier_directories(list(IDS), from_prepro_level=3,
                                          prepro_border=160,
                                          prepro_base_url=BASE_URL)
workflow.execute_entity_task(vas.terminus_profile_from_flowlines, gdirs)
for gd in gdirs:
    vas.mb_calibration_from_geodetic_mb(gd, overwrite_gdir=True, informed_threestep=True, temp_bias_file_path=URL)
    mbmod = vas.VAScalingMassBalance(gd)
    target = c._target_year(gd, mbmod, None)
    zmin, zmax = vas.get_min_max_elevation(gd)
    ref_new = c._reference_model(gd, mbmod, target)
    tp = ref_new.terminus_profile
    L0 = tp.length_m_ref
    ref_old = c._reference_model(gd, mbmod, target); ref_old.terminus_profile = None
    g = {'rgi_id': gd.rgi_id, 'name': IDS[gd.rgi_id], 'rgi_area': gd.rgi_area_km2,
         'target_yr': target, 'zmin': float(zmin), 'zmax': float(zmax), 'L0': L0 / 1e3,
         'params': {k: round(float(gd.settings[k]), 3) for k in ['melt_f', 'prcp_fac', 'temp_bias']}}
    print(gd.rgi_id, g['params'])

    # 1. terminus vs length
    Ls = np.linspace(0, 2.5 * L0, 120)
    g['profile'] = {'length': [round(x / 1e3, 4) for x in Ls],
                    'new': [round(tp(x), 1) for x in Ls],
                    'eq8': [round(float(zmax + x / L0 * (zmin - zmax)), 1) for x in Ls]}

    # 2. reconstructions
    rec = {}
    for name, ref in [('old', ref_old), ('new', ref_new)]:
        try:
            A = c.find_start_area_from_model(ref, YS, target_yr=target)
            rec[name] = series(c._start_model(ref, A, YS).run_until_and_store(YE))
        except Exception as e:
            print('  rec', name, 'failed', e); rec[name] = None
    # new, terminus anchored at the RGI date (simple fixed point)
    anchored = None
    for it in range(15):
        try:
            A = c.find_start_area_from_model(ref_new, YS, target_yr=target)
        except Exception:
            break
        ds = c._start_model(ref_new, A, YS).run_until_and_store(YE)
        z = float(ds.min_hgt.sel(time=target))
        if abs(z - zmin) < 1:
            anchored = series(ds); break
        tp.length_m_ref = float(ds.length_m.sel(time=target))
    tp.length_m_ref = L0
    rec['new_anchored'] = anchored
    print('  anchored converged:', anchored is not None)
    g['rec'] = rec

    # 3. 2004 area against 1901 area
    starts = np.geomspace(0.1, 80, 45)
    scan = {'start': [round(float(a), 4) for a in starts]}
    for name, ref in [('old', ref_old), ('new', ref_new)]:
        vals = []
        for a in starts:
            m = c._start_model(ref, a * 1e6, YS); m.run_until(target)
            vals.append(round(m.area_m2 / 1e6, 4))
        scan[name] = vals
    g['scan'] = scan

    # 4. trajectories from a few start areas
    traj = {'start_factor': [0.25, 0.5, 1, 2, 4]}
    for name, ref in [('old', ref_old), ('new', ref_new)]:
        traj[name] = []
        for f in traj['start_factor']:
            ds = c._start_model(ref, f * gd.rgi_area_m2, YS).run_until_and_store(YE)
            traj[name].append([round(float(x) / 1e6, 4) for x in ds.area_m2.values])
    g['traj'] = traj

    # 5. equilibrium start with a temperature bias
    dTs = np.round(np.arange(-2.0, 1.51, 0.05), 2)
    eq = {'dT': [], 'a1901': [], 'a_target': [], 'z_target': []}
    for dT in dTs:
        A = eq_area(mbmod, tp, dT)
        if A is None: continue
        m = c._start_model(ref_new, A, YS); m.run_until(target)
        eq['dT'].append(float(dT)); eq['a1901'].append(round(A / 1e6, 4))
        eq['a_target'].append(round(m.area_m2 / 1e6, 4)); eq['z_target'].append(round(m.min_hgt, 1))
    g['eq'] = eq
    out['glaciers'].append(g)

fp = os.path.join(WD, 'terminus_experiments.json')
with open(fp, 'w') as f:
    json.dump(out, f)
print('written to', fp)
