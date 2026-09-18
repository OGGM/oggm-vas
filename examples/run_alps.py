"""A full OGGM-VAS reconstruction: 1901-2020 from W5E5, through the RGI area.

The workflow is OGGM's, with two differences: the mass balance model is
`VAScalingMassBalance`, and the glacier evolves by volume/area scaling instead
of ice dynamics. No inversion is needed.

The glacier directories have to provide `gridded_data` (for the glacier
elevation range), `climate_historical` and `inversion_flowlines` -- OGGM's
calibration reads the flowlines even though the VAS model does not use them.
Prepro level 3 with elevation band flowlines gives all three, and gives one
flowline per glacier, which is what the VAS model expects.

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

    # Elevation band flowlines, W5E5 climate, level 3
    gdirs = workflow.init_glacier_directories(
        RGI_IDS, from_prepro_level=3, prepro_border=160,
        prepro_base_url=('https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                         'oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/'))

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

    # Reconstruct the whole record, passing through the observed area
    workflow.execute_entity_task(vascaling.run_reconstruction, gdirs, ys=ys,
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

# Note on the terminus elevation, which limits how far back this can go.
#
# The VAS model ties the terminus elevation to the glacier length through
#
#     min_hgt = max_hgt + (length / length_0) * (min_hgt_0 - max_hgt)
#
# anchored on the initial state. That relation has no physical basis outside
# the observed elevation range, and it is not bounded by it: a glacier
# reconstructed as several times its present size gets a terminus far below
# the lowest point of its DEM, and a glacier that shrinks a lot gets one close
# to its summit. Both feed straight back into the melt.
#
# The practical consequence is that for early start years the modelled mass
# change stops being monotonic in melt_f, so the dynamic calibration can fail
# to bracket the observations (it says so rather than returning a bad answer).
# Clipping the terminus to the glacier's DEM range restores monotonicity, but
# that is a change to the published model, so it is not done here.
