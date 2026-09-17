"""A minimal OGGM-VAS workflow: calibrate on geodetic mass balance, then run.

This is the same workflow as OGGM's, with two differences: the mass balance
model is `VAScalingMassBalance`, and the glacier evolves by volume/area
scaling instead of ice dynamics. No inversion is needed.

The glacier directories have to provide `gridded_data` (for the glacier
elevation range), `climate_historical` and `inversion_flowlines` -- OGGM's
calibration reads the flowlines even though the VAS model does not use them.
Prepro level 3 with elevation band flowlines gives all three, and gives one
flowline per glacier, which is what the VAS model expects.
"""
import os

import numpy as np

from oggm import cfg, utils, workflow
import oggm_vas as vascaling

# The glaciers to run
RGI_IDS = ['RGI60-11.00897',  # Hintereisferner
           'RGI60-11.00787',
           'RGI60-11.00746']


def run():
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

    # Calibrate melt_f, prcp_fac and temp_bias on Hugonnet et al. (2021),
    # through OGGM's own calibration task
    workflow.execute_entity_task(vascaling.mb_calibration_from_geodetic_mb,
                                 gdirs, overwrite_gdir=True)

    for gdir in gdirs:
        print('%s  melt_f %.3f  prcp_fac %.3f  temp_bias %.3f'
              % (gdir.rgi_id, gdir.settings['melt_f'],
                 gdir.settings['prcp_fac'], gdir.settings['temp_bias']))

    # Run over the calibration period
    workflow.execute_entity_task(vascaling.run_from_climate_data, gdirs,
                                 ys=2000, ye=2020, output_filesuffix='_vas')

    # The geometry is written to `model_diagnostics`, which OGGM's own
    # compilation tools understand
    ds = utils.compile_run_output(gdirs, input_filesuffix='_vas', path=False)
    print(ds.volume.to_pandas().round(0))

    # The VAS specific output (specific mass balance, terminus elevation,
    # response time scales) is in `vas_diagnostics`
    import xarray as xr
    with xr.open_dataset(gdirs[0].get_filepath('vas_diagnostics',
                                               filesuffix='_vas')) as vds:
        print(vds[['spec_mb', 'min_hgt', 'tau_l', 'tau_a']].to_dataframe())

    return gdirs


if __name__ == '__main__':
    run()
