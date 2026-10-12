![oggm-logo](docs/_static/logo.png)

[![Test](https://github.com/OGGM/oggm-vas/actions/workflows/run-tests.yml/badge.svg?branch=dev)](https://github.com/OGGM/oggm-vas/actions/workflows/run-tests.yml)

# OGGM-VAS

A python implementation of the volume/area scaling glacier model of
[Marzeion et al. (2012)](https://tc.copernicus.org/articles/6/1295/2012/),
running inside the [OGGM](https://oggm.org) workflow. Originally written by
[Moritz Oberrauch](https://github.com/oberrauch) for OGGM 1.4 (MSc thesis),
ported to current OGGM master in 2026.

This README is a handover note. It describes where the port stands
(October 2026), what is known to be wrong, and what to do next.

## Status in short

- Branch `dev` works against OGGM master (it needs the unreleased
  `gdir.settings` / `gdir.read_store` API, so OGGM 1.6.3 will not do).
- The mass balance model is calibrated by OGGM's own
  `mb_calibration_from_geodetic_mb` on Hugonnet et al. (2021), like any other
  OGGM mass balance model.
- The model runs a full 1901-2020 reconstruction that passes through the RGI
  area at the inventory date.
- **The reconstruction is not usable yet**, for two reasons that are
  described below and that interact:
  1. the terminus elevation of the glacier at the start of the run is wrong
     by construction, in this code and in the original MATLAB code alike;
  2. the reconstruction depends strongly on the temperature bias, which the
     geodetic calibration does not constrain.

## Getting started

    pip install git+https://github.com/OGGM/oggm.git   # OGGM master
    pip install -e ".[tests]"
    pytest oggm_vas                                    # 26 tests, ~40 s
    python examples/run_alps.py                        # three Alpine glaciers

The tests run on OGGM's sample data (Hintereisferner, HISTALP) and need no
other download. `examples/run_alps.py` downloads three prepro level 3 glacier
directories (~550 MB) and is the best place to start reading: it is the whole
workflow in one short function.

    import oggm_vas as vascaling
    vascaling.initialize()                 # cfg.initialize() + scaling params
    gdirs = workflow.init_glacier_directories(
        rgi_ids, from_prepro_level=3, prepro_border=160,
        prepro_base_url='.../oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5/')
    workflow.execute_entity_task(
        vascaling.mb_calibration_from_geodetic_mb, gdirs, overwrite_gdir=True)
    workflow.execute_entity_task(
        vascaling.run_reconstruction, gdirs, ys=1901, output_filesuffix='_vas')

Use elevation band directories. The model describes the glacier as a whole
and refuses glaciers with more than one flowline.

## What is in the package

Everything is in `oggm_vas/core.py`.

- `VAScalingMassBalance` subclasses OGGM's `MonthlyTIModel` and overrides one
  method, `_get_climate_for_index`. It ignores the heights it is given and
  returns the temperature at the terminus (`min_hgt`) and the solid
  precipitation computed from the `min_hgt` / `max_hgt` range (Sect. 2.1.1
  and 2.1.2 of the paper). Parameters, climate reading and the mass balance
  equation are inherited, which is why OGGM's calibration can drive it
  through `mb_model_class=VAScalingMassBalance` without any change in OGGM.
- `mb_calibration_from_geodetic_mb`: thin wrapper around OGGM's task.
- `VAScalingModel`: the scaling model itself (volume, area, length, terminus
  elevation, response times), one step per year.
- `find_start_area` / `run_reconstruction`: search for the area at the start
  year that gives the RGI area at the inventory date, then run the whole
  climate record.
- `run_from_climate_data`, `run_random_climate`, `run_constant_climate`:
  forward runs from the RGI state.
- `mb_calibration_dynamic_from_geodetic_mb`: fits `melt_f` against the mass
  change of the evolving glacier instead of the fixed RGI geometry. Probably
  to be dropped, see below.
- Output goes to two files: `model_diagnostics` (volume, area, length; read
  by `utils.compile_run_output`) and `vas_diagnostics` (adds `spec_mb`,
  `min_hgt`, `max_hgt`, `tau_l`, `tau_a`).

## What changed with respect to the 2012 model

These are deliberate, and follow OGGM:

- No t\*, no mu\*, no bias correction. `mu*` is OGGM's `melt_f`
  (kg m-2 day-1 K-1). `melt_f`, `prcp_fac` and `temp_bias` are calibrated per
  glacier on the 2000-2020 geodetic mass balance.
- Calendar years instead of hydrological years.
- No precipitation lapse rate. Precipitation is taken at the elevation of the
  climate data and scaled by `prcp_fac` only.
- Constant temperature lapse rate (`temp_default_gradient`, -6.5 K km-1)
  instead of a local regression on the CRU grid.
- OGGM's thresholds: `temp_melt` = -1 °C and `temp_all_solid` = 0 °C, against
  +1 °C and +3 °C in the paper. For this model that is the same as a
  temperature bias of about +2 K (melt) to +3 K (solid fraction) relative to
  the 2012 settings.
- Climate data is whatever the glacier directory holds (GSWP3-W5E5 in the
  standard prepro directories), not CRU.
- The turnover in the response time (Eq. 14) is the mean solid precipitation
  over the whole climate record, not the 31 years around t\*.

## Open problem 1: terminus elevation in the reconstruction

The model has one geometric feedback: Eq. (8) of the paper,

    z_term = z_max + (L / L_0) * (z_term_measured - z_max)

where `L_0` is, in the paper, the length in the year of the area measurement.
The reconstruction starts in 1901 from an unknown area and has to end up on
the RGI area at the inventory date. In the code (ours and the MATLAB one)
`L_0` is the length at the *start* of the run and `z_term_measured` is
applied to the start year. So the 1901 glacier always sits on today's
terminus elevation, however big it was, and the terminus moves away from the
observed elevation as the run approaches the inventory date.

What this does, from `examples/run_alps.py` (W5E5, 1901 start, RGI date 2003):

| glacier | area 1901 / 2004 (km2) | terminus 1901 / 2004 (m) | Hugonnet 2000-2020 | modelled 2000-2020 |
| --- | --- | --- | --- | --- |
| RGI60-11.00897 (Hintereisferner) | 8.82 / 8.04 | 2431 / 2481 | -1100 | -767 |
| RGI60-11.00787 | 3.14 / 3.97 | 2779 / 2671 | -515 | -882 |
| RGI60-11.00746 | 16.98 / 16.62 | 2144 / 2151 | -708 | -647 |

Mass change is in kg m-2 yr-1 on the RGI area. The 1901 terminus is the
observed (RGI) one. Three symptoms:

1. The terminus at the inventory date is not the observed one (50 m too high
   for Hintereisferner, 108 m too low for 11.00787).
2. So the reconstruction does not reproduce the geodetic mass balance it was
   calibrated on. The calibration uses the fixed RGI geometry and closes
   exactly; the evolving glacier is off by up to 70 %.
3. The glacier change over the century is far too small. Hintereisferner
   loses 10 % of its area since 1901 where Fig. 13 of the paper has about
   11.3 km2 in 1920 and 10.5 km2 in 1940. One glacier is *smaller* in 1901
   than today.

### What Ben Marzeion says (October 2026)

Ben knows the problem. Moritz ran into it during the thesis, and Ben's
current MATLAB code has it too ("moved, not solved"). What he tried:

- Anchoring `L_0` at the inventory date properly needs the model length at
  that date, which depends on the run itself because of the response time. So
  it takes another iteration around the start-area search, which was too
  expensive in MATLAB.
- Taking `L_0` from the scaling law at the inventory date, as if the glacier
  were in equilibrium then: the glacier comes out too short, hence too steep,
  and the iteration becomes unstable (steep glacier -> much lower terminus in
  1901 -> much more melt).
- His idea for a real fix: replace the straight line of Eq. (8) by a simple
  function for the bed profile, steep at the top and flat at the bottom
  (Oerlemans uses an exponential; doi 10.1016/j.mex.2020.10117, as quoted in
  the chat and possibly truncated, uses a sum of two exponentials), fitted on
  RGI and ice thickness data.

OGGM has what this needs: the flowline of each glacier and the downstream
line below the tongue (`downstream_line`, key `surface_h`), so the terminus
elevation for a given length can be read off real terrain.

### What was tried here

An earlier experiment (September 2026) on Hintereisferner looked at two
changes. The code was not kept, so the numbers below are from notes and have
to be reproduced before they are relied on.

- Anchoring Eq. (8) at the inventory date, as the paper says. Unbounded it
  runs away, exactly as Ben describes: the slope implied by Eq. (8) for
  Hintereisferner is about 26 %, the terrain below the tongue is about 10 %.
  With a floor at the lowest ice (2431 m) it behaves. A floor at the bottom
  of the downstream line (1945 m) overshoots badly.
- A temperature bias: OGGM's own value for this glacier (+1.76 K), with
  `melt_f` re-fitted to the same target. See open problem 2.

Both together gave 11.6 km2 in 1901, 11.4 in 1920, 10.8 in 1940 and
-1027 kg m-2 yr-1 for 2000-2020, close to the observations. Neither alone was
enough.

## Open problem 2: the temperature bias decides the reconstruction

Hugonnet gives one number per glacier and the model has three parameters.
For any `temp_bias` there is a `melt_f` that reproduces the 2000-2020 mass
balance exactly, and these parameter pairs are indistinguishable over the
calibration period. They are not the same model before it. A warm bias
lengthens the melt season and needs a smaller `melt_f`, so the mass balance
reacts less to a given temperature change, and the cool early century ends up
closer to the recent, negative, value. A cold bias does the opposite.

This is new with the geodetic calibration. In 2012 `mu*` came from requiring
zero mass balance in a 31 year climate around t\*, somewhere in the 20th
century, and there was no temperature bias. The long-term mean was pinned and
the recent decades were the extrapolation. Now the recent decades are pinned
and the whole century before them is the extrapolation.

Hintereisferner, `prcp_fac` 3.53, `melt_f` fitted on the fixed RGI geometry
for each prescribed `temp_bias`, reconstruction from 1901 (RGI area 8.04 km2,
all rows reproduce -1100 kg m-2 yr-1 on fixed geometry):

| `temp_bias` (K) | `melt_f` | MB 1901-1930, fixed geometry | area 1901 (km2) | area 1940 (km2) | modelled 2000-2020 |
| --- | --- | --- | --- | --- | --- |
| -2 | 6.01 | +294 | 8.09 | 8.69 | -990 |
| -1 | 4.46 | +98 | 8.39 | 8.60 | -875 |
| 0 | 3.38 | -69 | 8.82 | 8.68 | -767 |
| +1 | 2.63 | -209 | 9.43 | 8.95 | -680 |
| +1.76 (OGGM) | 2.20 | -303 | 10.04 | 9.31 | -627 |
| +2 | 2.09 | -329 | 10.25 | 9.45 | -614 |
| +3 | 1.68 | -426 | 11.17 | 10.08 | -575 |

So over a plausible range of biases the glacier has lost anything between
nothing and almost 30 % of its area since 1901, 0.3 to 0.9 km2 per K. The other two
test glaciers behave the same way: 3.0 to 4.0 km2 in 1901 for RGI60-11.00787
(3.97 today) and 15.3 to 19.4 km2 for RGI60-11.00746 (16.62 today) for biases
of -3 to +3 K.

The last column shows how this couples to problem 1. The larger the 1901
glacier, the more it shrinks, the higher the terminus has climbed above the
observed one by 2004 (2555 m at +1.76 K, against 2431 m), and the further
the evolving glacier is from the mass balance it was calibrated on.

Fitting `melt_f` on the evolving glacier instead
(`mb_calibration_dynamic_from_geodetic_mb`) does not help while problem 1 is
there. It makes the sensitivity explode, and for biases between about -0.5
and +1 K there is no solution at all (`melt_f` ends on a bound):

| `temp_bias` (K) | -3 | -2 | -1 | -0.5 to +1 | +1.5 | +1.76 | +2 | +3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `melt_f` | 7.95 | 9.14 | 13.2 | at a bound | 11.8 | 9.75 | 8.19 | 4.48 |
| area 1901 (km2) | 7.7 | 10.3 | 17.5 | - | 200 | 152 | 121 | 63 |

`examples/run_hef_temp_bias.py` produces this scan and the figure
`examples/RGI60-11.00897_temp_bias_scan.png`.

What to do about it is not decided. The options:

- Use OGGM's temperature bias and move on. OGGM has the same degeneracy and
  settles it with a prior per climate grid cell (`informed_threestep`). It is
  the pragmatic choice and keeps the two models comparable. The caveat is
  that the prior was derived with OGGM's flowline model and OGGM's
  thresholds; this model melts at terminus temperature, and nothing says the
  same bias is right for it.
- Constrain the bias with something that sees the past: observed length or
  area changes (Fig. 13 of the paper is such a comparison), or long in-situ
  mass balance series where they exist.
- Carry it as an uncertainty, with a small ensemble over `temp_bias`.

Two related things were not looked at. `prcp_fac` has not been scanned; it
should act the same way (more precipitation needs a larger `melt_f`, hence
more sensitivity), but that is an expectation, not a result. And `temp_melt`
and `temp_all_solid` are equivalent to a temperature bias in this model, so
the choice of OGGM's thresholds over the 2012 ones is part of the same
question.

## Comparison with Ben's MATLAB code

Ben's current script (`matlab-code/model_cru_hugonnet.m`: CRU forcing,
calibrated on Hugonnet) was compared with `core.py` line by line.
`examples/compare_matlab_stepping.py` transliterates its time loop with a
switch for each difference. With all switches on the python side it
reproduces `VAScalingModel` to machine precision, so the list below is
complete as far as the time stepping goes.

Same in both:

- scaling constants and exponents, for glaciers and ice caps;
- start state from the scaling laws, order of operations in a step
  (mass balance, volume, then area and length relaxed with time scales taken
  from the state at the start of the year);
- Eq. (8) with `L_0` = length at the start of the run and the RGI terminus
  applied to the start year. `adjust_term_elev=False`, our default, is
  exactly the MATLAB behaviour;
- turnover and time scales limited to 10 mm w.e. yr-1 and 1 yr;
- calibration on the fixed RGI geometry over 2000-2019, no t\*, no bias term;
- start area found by iterating until the area at the RGI year matches.

Different in the time stepping. The last column is the effect on the 1901
area of Hintereisferner in the offline test setup (HISTALP, made-up
calibration; reference 10.16 km2), so read it as an order of magnitude:

| | MATLAB | python | effect |
| --- | --- | --- | --- |
| solid precipitation | computed once on the RGI geometry | follows the moving terminus (as in the paper) | +1.1 % |
| volume change (l. 282) | `B * A`, no ice density | `B * A / 900` (Eq. 7) | -1.8 % |
| `tau_A` (l. 285) | `tau_L * A/L²` with A and L from the scaling of V | with the actual A and L (Eq. 15) | -0.4 % |
| turnover | mean solid precipitation 2000-2019 | whole climate record | < 0.1 % |
| melt factor | per month | per day (OGGM) | < 0.1 % |
| terminus floor (l. 271) | not below sea level | none | none here |
| volume reaches 0 | area and length set to 0 at once | relaxed over the next step | none here |
| all together | | | -1.3 % |

None of these explains the weak reconstruction. The missing ice density in
the MATLAB script is worth raising with Ben.

Different in inputs and parameters (not tested numerically):

- climate: CRU TS 4.06 anomalies on the CRU CL 2.0 climatology, with the
  precipitation factor applied to the climatology only; we use the glacier
  directory's climate with `prcp_fac` on the total;
- temperature lapse rate from a regression over the surrounding CRU cells;
  ours is constant;
- precipitation lapse rate, to the mean of the RGI `zmin` and `zmax`; ours
  has none;
- `temp_melt`, `temp_all_solid`, the precipitation factor and the
  precipitation lapse rate are global and set per experiment
  (`load_parameters`, not in the script); ours are OGGM's;
- only `mu` is calibrated, in closed form; we use OGGM's three-step
  calibration with bounds on `melt_f`;
- hydrological years (October-September, April-March in the south); ours are
  calendar years;
- start-area search: 1 % tolerance, 50 iterations, glaciers that fail are
  dropped from the output; ours is a bracketed root search to 0.01 % that
  raises on failure;
- RGI year clipped to 1903-2020.

## Next steps

1. Decide how the terminus elevation depends on length. Preferred: read it
   from the OGGM flowline and downstream line instead of Eq. (8). Open points:
   how the scaling length maps on the flowline length (for Hintereisferner
   4.9 km from scaling, 5.6 km for the elevation band flowline), and what to
   do with `max_hgt`, which is constant today and wrong for a glacier that
   shrinks a lot.
2. Anchor the terminus at the inventory date, so the run passes through the
   observed area *and* the observed terminus. The outer iteration Ben found
   too expensive costs milliseconds here.
3. Decide how to set the temperature bias (open problem 2). The quick route
   is OGGM's `informed_threestep`, which can be passed through
   `mb_calibration_from_geodetic_mb` but has not been run with this model.
4. Re-check the three Alpine glaciers against Fig. 13 of the paper.
5. Decide on `mb_calibration_dynamic_from_geodetic_mb`. With the current
   anchoring the modelled mass change is not monotonic in `melt_f` and the
   fit can fail. If steps 1-2 make the reconstruction reproduce the geodetic
   mass balance well enough, drop it.
6. A fallback for glaciers where the start-area search fails. We raise; the
   2012 paper substituted regional mean rates (Sect. 6.2.2), the MATLAB
   script drops them. A global run needs one.
7. Only then: glacier directories with the 1901-2020 reconstruction.

## Things that will bite

- `init_glacier_directories(from_prepro_level=3)` re-extracts the archive on
  every call and restores OGGM's own calibration, wiping the VAS one.
  Re-open existing directories without `from_prepro_level`.
- `gdir.settings` falls back to the `mb_calib.json` of the prepro directory.
  An uncalibrated VAS glacier silently runs with OGGM's flowline parameters.
- OGGM's calibration reads `inversion_flowlines` even though the VAS model
  does not use them, and wraps the model in `MultipleFlowlineMassBalance`
  when there is more than one flowline. Hence elevation band directories.
- `override_missing` in OGGM's geodetic calibration does not help when the
  period is absent from the Hugonnet table.
- `utils.compile_run_output` rejects unknown variables and reads
  `water_level`, `glen_a` and `fs` unconditionally, which is why the VAS
  variables live in a separate file and those three are written as NaN.
- The package tracks OGGM master and breaks when the API moves (last time:
  `read_pickle` -> `read_store`, October 2026). CI installs OGGM master.

## People

- Ben Marzeion: the model and the MATLAB code.
- Moritz Oberrauch: the original python implementation and thesis.
- Fabien Maussion: the port to OGGM master.

Source and issues: [github.com/OGGM/oggm-vas](https://github.com/OGGM/oggm-vas).
