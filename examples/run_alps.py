"""A full OGGM-VAS reconstruction: 1901-2020 from W5E5, through the RGI area.

The workflow is OGGM's, with two differences: the mass balance model is
`VAScalingMassBalance`, and the glacier evolves by volume/area scaling instead
of ice dynamics. No inversion is needed.

The glacier directories have to provide `gridded_data` (for the glacier
elevation range), `climate_historical`, and `inversion_flowlines` plus
`downstream_line` for the terminus profile. Prepro level 3 gives all of
them. We use elevation bands: their flowline length is close to the
scaling length of the VAS model and their top close to the glacier top,
which is what the terminus profile needs.

The key step is `run_reconstruction`: it picks the glacier area at `ys` so
that the model reproduces the RGI area at the inventory date, then carries on
to the end of the climate record. Since the inventory date ranges from the
1960s to the 2010s across the RGI, this is what makes glaciers comparable to
each other -- each passes through its own observed area at its own observed
date. `run_from_climate_data` instead imposes the RGI area at `ys` whatever
year that is, which is only right when `ys` is the inventory date.
"""
import os

from oggm import cfg, utils, workflow
import oggm_vas as vascaling

# The glaciers to run
RGI_IDS = ['RGI60-11.00897',  # Hintereisferner
           'RGI60-11.00787',
           'RGI60-11.00746']

# W5E5 starts in 1901
YS = 1901


def run(ys=YS, dynamic_calibration=False):
    """Calibrate and reconstruct.

    Parameters
    ----------
    ys : int
        the year to reconstruct from
    dynamic_calibration : bool
        if True, calibrate melt_f against the mass change of the *evolving*
        glacier rather than of the fixed RGI geometry. This is the more
        correct thing to do, but see the note at the bottom of this file:
        for early start years the terminus elevation parameterisation can
        push the solution outside the range where it makes sense, and the
        calibration then reports that it could not match the observations.
    """
    # `vascaling.initialize` calls `cfg.initialize` and adds the scaling
    # parameters on top
    vascaling.initialize(logging_level='WORKFLOW')
    cfg.PATHS['working_dir'] = utils.mkdir(
        os.path.join(os.path.expanduser('~'), 'OGGM', 'vas_alps'))

    # Elevation bands, W5E5 climate, level 3. The border of 160 grid points
    # gives a downstream line long enough for the large past glaciers.
    gdirs = workflow.init_glacier_directories(
        RGI_IDS, from_prepro_level=3, prepro_border=160,
        prepro_base_url=('https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                         'oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/'))

    # The terminus moves along the main flowline and the downstream line
    # rather than along the straight line of Eq. (8) of Marzeion et al. (2012)
    workflow.execute_entity_task(vascaling.terminus_profile_from_flowlines,
                                 gdirs)

    # Calibrate melt_f, prcp_fac and temp_bias on Hugonnet et al. (2021)
    if dynamic_calibration:
        task = vascaling.mb_calibration_dynamic_from_geodetic_mb
        kwargs = dict(ys=ys)
    else:
        task = vascaling.mb_calibration_from_geodetic_mb
        kwargs = dict()
    workflow.execute_entity_task(task, gdirs, overwrite_gdir=True, **kwargs)

    for gdir in gdirs:
        print('%s  melt_f %.3f  prcp_fac %.3f  temp_bias %.3f'
              % (gdir.rgi_id, gdir.settings['melt_f'],
                 gdir.settings['prcp_fac'], gdir.settings['temp_bias']))

    # Reconstruct the whole record, passing through the observed area. With
    # the terrain profile several 1901 areas can reproduce it (three for
    # Kesselwandferner, README open problem 3); take the smallest for now.
    workflow.execute_entity_task(vascaling.run_reconstruction, gdirs, ys=ys,
                                 multiple_solutions='smallest',
                                 output_filesuffix='_vas')

    # The geometry goes to `model_diagnostics`, which OGGM's own compilation
    # tools understand
    ds = utils.compile_run_output(gdirs, input_filesuffix='_vas', path=False)
    print(ds.volume.to_pandas().round(0))

    # Check that each glacier really does pass through its RGI area
    print('\ngeometry match at the inventory date:')
    for gdir in gdirs:
        target_yr = gdir.get_diagnostics()['vas_reconstruction_target_yr']
        area = float(ds.area.sel(rgi_id=gdir.rgi_id, time=target_yr))
        print('  %s  %d: modelled %.4f km2, RGI %.4f km2'
              % (gdir.rgi_id, target_yr, area / 1e6, gdir.rgi_area_km2))

    # The VAS specific output (specific mass balance, terminus elevation,
    # response time scales) is in `vas_diagnostics`
    import xarray as xr
    with xr.open_dataset(gdirs[0].get_filepath('vas_diagnostics',
                                               filesuffix='_vas')) as vds:
        print(vds[['spec_mb', 'min_hgt', 'tau_l', 'tau_a']].to_dataframe())

    return gdirs


if __name__ == '__main__':
    run()

# Note on the terminus elevation.
#
# With the profile written by `terminus_profile_from_flowlines`, the change in
# VAS length since the RGI date is applied along the main flowline and the
# downstream line, and the terminus sits at the surface elevation there. It
# does not extrapolate past the end of the downstream line. Without the
# profile, the model falls back to Eq. (8) of Marzeion et al. (2012),
#
#     min_hgt = max_hgt + (length / length_0) * (min_hgt_0 - max_hgt)
#
# anchored on the start year, which is not bounded by the terrain and puts
# the terminus of a large past glacier far below the valley floor.
#
# With the profile the terminus feeds back on the mass balance as it should,
# and the start-area search is no longer well posed: several 1901 areas can
# end on the RGI area, and the terminus at the RGI date is not the observed
# one. See "Open problem 3" in the README and examples/terminus_experiments.py.
