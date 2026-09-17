## Code rules

1. Ask, don't assume. For anything nontrivial or ambiguous, ask before writing code — don't make silent assumptions about intent, architecture, or requirements. For small, unambiguous fixes (typos, renames, obvious one-liners), just do it.
2. If you see a clearly better approach, say so before implementing. Explain the tradeoff in 2-4 bullets.
3. Flag uncertainty explicitly. If you're not confident about an approach or technical detail, say so plainly before proceeding — "I'm not sure X is true, worth checking" — not a performance of being uncertain. Confidence without certainty causes more damage than admitting a gap.

## Git + github

- Do not create or switch git branches without my approval. Use internal cached git if you need it.
- Never push anything to github. You can fetch if you want.

## Communication style

- Write plainly, like an engineer or scientist leaving a code comment or Slack message, not a blog post.
- No section headers, no bolded phrase-titles, no dramatic framing ("blast radius", "being straight about", "the reckoning").
- Don't invent terminology or metaphors for ordinary concepts. If the codebase or field already has a name for it, use that name.
- Default to short answers. State what changed and why in 1-3 sentences. If an explanation genuinely needs more than ~5-6 sentences, give the short version first and offer to expand.
- Skip the narration of your own reasoning process. No self-congratulation or self-flagellation about earlier mistakes — just state the correction.
- I am a scientist, not a CS graduate. Avoid jargon when it becomes too heavy. But at the same time I wrote most of OGGM, so I do know some coding.

## Python environment

Use /Users/uu23343/.mambaforge/envs/oggm_env_314/bin/python if possible. It has latest OGGM installed.

## This project

This codebase is a python implementation of the Marzeion 2012 model (paper found in ./context/) into python for the OGGM framework. The reimplementation was for the MSc Thesis (thesis found in ./context/), and is 5 years old at least. It was made compatible with OGGM 1.4 back in the days, which is very different from current oggm master (code found in /Users/uu23343/Library/CloudStorage/Dropbox/HomeDocs/git/oggm-mac). You are allowed to look at this folder and OGGM's.

The overarching goal of the work will be to make OGGM VAS compatible with latest OGGM. Importantly, OGGM-VAS will need to calibrate the three parameters (melt_f, temp_bias, prcp_fac) the same way as OGGM does, ideally using the same calibration_from_geodetic_mb routine.

Achieving this probably includes:

- adapting OGGM-VAS to the new ways to handle parameters, etc.
- make OGGM-VAS work with the new calibration
- if needed, make some upstream changes in OGGM to allow for some oggm-vas quirks. if these changes in OGGM are too consequent, we copy-paste what we need here and make the changes here (we override the upstream function).

### Current state (Sep 2026)

The package does not import against current OGGM: `entity_task(writes=['climate_info'])`
raises `KeyError: 'climate_info'` at import time. `oggm_vas/core.py` is written for
OGGM 1.4, and OGGM deleted the t*/mu* calibration architecture in v1.6.0 (Mar 2023,
commit 410bc787).

Target is the local OGGM master checkout at ../oggm-mac (v1.6.3-81), which is ahead of
the 1.6.3 release and includes the unreleased `gdir.settings` / `gdir.observations`
refactor. That means: tasks read `gdir.settings[...]` rather than `cfg.PARAMS[...]`
(settings falls back to cfg.PARAMS as parent), calibration results are written to
`settings.yml` rather than `mb_calib.json`, and every task carries
`settings_filesuffix=` / `observations_filesuffix=`.

What changed upstream that matters here:

- gone: `t_star`, `mu_star`, `local_t_star`, `mu_star_calibration`, `ref_tstars.csv`,
  the `climate_info` basename, `PastMassBalance`, `robust_model_run`,
  `utils.monthly_timeseries`, `AvgClimateMassBalance`. `oggm.core.climate` now holds only
  three tasks; the climate processors moved to `oggm/shop/`.
- gone from cfg.PARAMS: `prcp_scaling_factor` (-> `prcp_fac`), `run_mb_calibration`,
  `mu_star_halfperiod`, `min_mu_star`/`max_mu_star` (-> `melt_f_min`/`melt_f_max`),
  `tstar_search_window`, `use_bias_for_run`, `temp_local_gradient_bounds`,
  `climate_qc_months`.
- renamed: `MassBalanceModel.rho` -> `.ice_density`; the `SEC_IN_YEAR`/`SEC_IN_MONTH`
  constants -> `self.sec_in_year(year)` / `self.days_in_month(year)` methods.
- `climate_historical.nc` no longer carries a `gradient` variable; a constant
  `temp_default_gradient` is used instead.
- OGGM is calendar-year throughout, enforced by `MonthlyTIModel._check_for_full_years`.
  oggm-vas is hydro-year end to end today; it has to follow OGGM.

Design decisions for the port:

- `VAScalingMassBalance` subclasses `MonthlyTIModel` (not `MassBalanceModel`) and
  overrides only `_get_climate_for_index(heights, pok)`, ignoring `heights` and computing
  the Marzeion terminus temperature and solid-precipitation fraction from `min_hgt` /
  `max_hgt`. Everything above that seam is inherited, which is what makes the class
  pluggable into OGGM's calibration.
- Calibration goes through OGGM's
  `mb_calibration_from_geodetic_mb(..., mb_model_class=VAScalingMassBalance)`.
  Its duck type requires: constructor kwargs `gdir/melt_f/temp_bias/prcp_fac/
  check_calib_params/settings_filesuffix`, settable `melt_f`/`prcp_fac`/`temp_bias`,
  `is_year_valid` + `ys_float`/`ye_float`, `get_specific_mb`, `filename` /
  `input_filesuffix`, and the four MB_GLOBAL_PARAMS attributes. Two landmines: the
  unconditional `gdir.read_pickle('inversion_flowlines')` at massbalance.py:4920, and the
  silent wrap in `MultipleFlowlineMassBalance` when a glacier has more than one flowline
  (massbalance.py:5044) — so use elevation-band prepro L3 directories.
- `mu_star` becomes OGGM's `melt_f` (kg m-2 day-1 K-1, day-weighted). `prcp_fac` and
  `temp_bias` become calibrated per-glacier parameters instead of a global constant and 0.
- The precipitation lapse rate is dropped, matching OGGM's view that it adds noise rather
  than value: no `prcp_default_gradient`, no `prcp_grad` term in `compute_solid_prcp`.
  Precipitation is used at the climate file's `ref_hgt`, scaled only by `prcp_fac`.
  `min_hgt`/`max_hgt` still set the solid fraction.
- The t*/mu* machinery and the `vas_ref_tstars_*` data files get deleted, not ported.
- `prcp_clim` (the turnover in the response-time formula, and the last hidden dependency
  on t*) is redefined as the mean solid precipitation over the calibration reference period.
- The workflow assumes standard OGGM prepro L3 glacier directories.
