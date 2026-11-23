"""Hintereisferner 1901-2020, one reconstruction per prescribed temp_bias.

`temp_bias` is scanned on a regular grid (-3 to +3 K, every 0.2 K). For each
value `melt_f` is calibrated with the *dynamic* geodetic calibration, i.e.
against the mass change of the evolving glacier, so every run reproduces the
Hugonnet et al. (2021) mass change and the RGI area at the inventory date.
`prcp_fac` is not calibrated here: it is OGGM's own value for this glacier,
read from the prepro directory.

What the scan shows is how much of the reconstruction is actually constrained
by the geodetic observation: all the lines agree over the calibration period
by construction, and they fan out going back in time. A warm bias lifts the
melt more in the cold early century than in the recent decades (the melt
threshold is non-linear), so it needs a smaller `melt_f` to hit the same
target, and gives a bigger glacier in 1901.
"""
import os
import logging

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from oggm import cfg, utils, workflow
import oggm_vas as vascaling

log = logging.getLogger(__name__)

# Hintereisferner
RGI_ID = 'RGI60-11.00897'

# W5E5 starts in 1901
YS = 1901

# the temperature biases to prescribe [K]
TEMP_BIASES = np.round(np.arange(-3, 3.01, 0.2), 2)


def run(rgi_id=RGI_ID, ys=YS, temp_biases=TEMP_BIASES):
    """Calibrate and reconstruct once per prescribed temperature bias."""

    vascaling.initialize(logging_level='WORKFLOW')
    cfg.PATHS['working_dir'] = utils.mkdir(
        os.path.join(os.path.expanduser('~'), 'OGGM', 'vas_hef_temp_bias'))

    # Elevation band flowlines, W5E5 climate, level 3
    gdir = workflow.init_glacier_directories(
        [rgi_id], from_prepro_level=3, prepro_border=160,
        prepro_base_url=('https://cluster.klima.uni-bremen.de/~oggm/gdirs/'
                         'oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/'))[0]

    # OGGM's own precipitation factor for this glacier, from the prepro
    # calibration. Read it before anything of ours writes to the settings.
    prcp_fac = gdir.settings['prcp_fac']
    print('%s: OGGM prcp_fac %.3f' % (gdir.rgi_id, prcp_fac))

    runs = dict()
    mismatch = dict()
    for temp_bias in temp_biases:
        suffix = '_tb%+.1f' % temp_bias
        try:
            # calibrate melt_f against the evolving glacier, with temp_bias
            # and prcp_fac held fixed. Nothing is written to the gdir, so
            # every iteration starts from the same state.
            df = vascaling.mb_calibration_dynamic_from_geodetic_mb(
                gdir, ys=ys, temp_bias=temp_bias, prcp_fac=prcp_fac,
                write_to_gdir=False)
            # the reconstruction with those parameters
            vascaling.run_reconstruction(
                gdir, ys=ys, output_filesuffix=suffix,
                melt_f=df['melt_f'], prcp_fac=prcp_fac, temp_bias=temp_bias,
                check_calib_params=False)
        except Exception as err:
            # a bias far from the optimum can ask for a start glacier we
            # cannot reconstruct - skip it rather than stop the scan
            print('  temp_bias %+.1f: skipped (%s: %s)'
                  % (temp_bias, type(err).__name__, err))
            continue

        with xr.open_dataset(gdir.get_filepath('vas_diagnostics',
                                               filesuffix=suffix)) as ds:
            runs[temp_bias] = ds.load()
        mismatch[temp_bias] = df['vas_dmdtda_mismatch']

        print('  temp_bias %+.1f  melt_f %6.3f  mismatch %7.1f kg m-2 yr-1  '
              '%d area %.2f km2' % (temp_bias, df['melt_f'],
                                    df['vas_dmdtda_mismatch'], ys,
                                    runs[temp_bias].area_m2[0] / 1e6))

    if not runs:
        raise RuntimeError('no temperature bias could be reconstructed')

    # next to this script
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '%s_temp_bias_scan.png' % rgi_id)
    plot(runs, gdir, fpath, mismatch=mismatch)
    print('figure written to %s' % fpath)

    return gdir, runs


def plot(runs, gdir, fpath=None, mismatch=None, mismatch_tol=1):
    """One line per temperature bias, four panels.

    Runs where the calibration could not match the geodetic mass balance -
    `melt_f` ended up pinned to a bound - are drawn dashed: they are not
    constrained by the observation and should not be read as such.
    """

    biases = np.array(sorted(runs))
    # temp_bias is a signed quantity around zero, so a diverging ramp
    # centred on no bias
    norm = plt.Normalize(-max(abs(biases)), max(abs(biases)))
    cmap = plt.get_cmap('RdBu_r')

    panels = [('volume_m3', 1e9, 'Volume (km$^3$)'),
              ('area_m2', 1e6, 'Area (km$^2$)'),
              ('length_m', 1e3, 'Length (km)'),
              ('min_hgt', 1, 'Terminus elevation (m a.s.l.)')]

    fig, axs = plt.subplots(len(panels), 1, figsize=(9, 11), sharex=True)

    for ax, (var, scale, label) in zip(axs, panels):
        for temp_bias in biases:
            ds = runs[temp_bias]
            failed = (mismatch is not None
                      and abs(mismatch.get(temp_bias, 0)) > mismatch_tol)
            ax.plot(ds.time, ds[var] / scale, lw=1.2,
                    ls='--' if failed else '-',
                    color=cmap(norm(temp_bias)))
        ax.set_ylabel(label)
        ax.grid(True, color='0.9', lw=0.6)
        ax.set_axisbelow(True)

    # the observation every run is made to pass through
    target_yr = gdir.get_diagnostics()['vas_reconstruction_target_yr']
    axs[1].plot(target_yr, gdir.rgi_area_km2, 'ko', ms=6, zorder=5,
                label='RGI area at the inventory date')
    handles = axs[1].get_legend_handles_labels()[0]
    if mismatch is not None and any(abs(m) > mismatch_tol
                                    for m in mismatch.values()):
        handles.append(plt.Line2D([], [], color='0.4', ls='--', lw=1.2,
                                  label='geodetic MB not matched'))
    axs[1].legend(handles=handles, loc='upper right', frameon=False)

    axs[-1].set_xlabel('Year')
    axs[0].set_title('%s: reconstruction for a prescribed temperature bias\n'
                     '(melt_f calibrated on the geodetic mass balance for '
                     'each bias, prcp_fac = %.2f from OGGM)'
                     % (gdir.rgi_id, gdir.settings['prcp_fac']))

    fig.subplots_adjust(right=0.86, hspace=0.08)
    cax = fig.add_axes([0.89, 0.11, 0.02, 0.77])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label='Prescribed temperature bias (K)')

    if fpath is not None:
        fig.savefig(fpath, dpi=150)
    return fig


if __name__ == '__main__':
    run()
    plt.show()
