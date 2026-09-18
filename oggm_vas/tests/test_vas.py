"""Tests for the volume/area scaling model in `core.py`."""

# External libs
import os
import copy
import shutil

import numpy as np
import xarray as xr

# import test libs
import unittest
import pytest

# import OGGM modules
import oggm
import oggm.cfg as cfg
from oggm import utils
from oggm.utils import get_demo_file, rmsd_bc, rel_err, corrcoef
from oggm.core import (gis, climate, centerlines, massbalance, flowline,
                       inversion)
from oggm.tests.funcs import get_test_dir
import oggm_vas as vascaling

# import gis libs
gpd = pytest.importorskip('geopandas')

# The HISTALP demo file covers 1801-2002 for Hintereisferner. Calibrating on
# a made up geodetic mass balance over this period keeps the tests offline.
REF_MB = -500.
REF_MB_YEARS = (1953, 2003)


class TestVAScalingModel(unittest.TestCase):

    def setUp(self):
        # test directory
        self.testdir = os.path.join(get_test_dir(), 'tmp_vas')
        self.clean_dir()

        # load default parameter file
        vascaling.initialize()
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PARAMS['use_multiprocessing'] = False
        cfg.PARAMS['use_intersects'] = False
        cfg.PATHS['dem_file'] = get_demo_file('hef_srtm.tif')
        cfg.PARAMS['baseline_climate'] = 'CUSTOM'
        cfg.PATHS['climate_file'] = get_demo_file('histalp_merged_hef.nc')
        # HISTALP is not supported by `decide_winter_precip_factor`, so the
        # precipitation factor needs an explicit starting value
        cfg.PARAMS['prcp_fac'] = 2.5
        # the VAS model describes the glacier as a whole: one flowline only
        cfg.PARAMS['use_multiple_flowlines'] = False
        cfg.PARAMS['border'] = 40

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir, ignore_errors=True)
        os.makedirs(self.testdir)

    # -- helpers

    def _gdir(self, flowlines=True):
        """Glacier directory for Hintereisferner with climate data."""
        hef_file = get_demo_file('Hintereisferner_RGI6.shp')
        entity = gpd.read_file(hef_file).iloc[0]
        gdir = oggm.GlacierDirectory(entity, base_dir=self.testdir)

        gis.define_glacier_region(gdir)
        gis.glacier_masks(gdir)
        if flowlines:
            centerlines.compute_centerlines(gdir)
            centerlines.initialize_flowlines(gdir)
            centerlines.catchment_area(gdir)
            centerlines.catchment_intersections(gdir)
            centerlines.catchment_width_geom(gdir)
            centerlines.catchment_width_correction(gdir)
        climate.process_custom_climate_data(gdir)
        return gdir

    def _calibrated_gdir(self, ref_mb=REF_MB, **kwargs):
        """Glacier directory with calibrated VAS mass balance parameters."""
        gdir = self._gdir()
        massbalance.mb_calibration_from_scalar_mb(
            gdir, ref_mb=ref_mb, ref_mb_years=REF_MB_YEARS,
            mb_model_class=vascaling.VAScalingMassBalance,
            calibrate_param1='melt_f', calibrate_param2='prcp_fac',
            calibrate_param3='temp_bias', overwrite_gdir=True,
            overwrite_observations=True, **kwargs)
        return gdir

    def _vas_model(self, gdir=None, y0=None):
        """A `VAScalingModel` of Hintereisferner, ready to run."""
        if gdir is None:
            gdir = self._calibrated_gdir()
        mbmod = vascaling.VAScalingMassBalance(gdir)
        min_hgt, max_hgt = vascaling.get_min_max_elevation(gdir)
        if y0 is None:
            y0 = gdir.get_climate_info()['baseline_yr_0']
        model = vascaling.VAScalingModel(year_0=y0,
                                         area_m2_0=gdir.rgi_area_m2,
                                         min_hgt=min_hgt, max_hgt=max_hgt,
                                         mb_model=mbmod,
                                         glacier_type=gdir.glacier_type)
        return gdir, model

    # -- climate helpers

    def test_terminus_temp(self):
        """Testing the subroutine which computes the terminus temperature
        from the given climate file and glacier DEM.
        """
        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]

        temp_anomaly = 0
        temp_grad = -0.0065

        # the terminus temperature must equal the input temperature
        # if terminus elevation equals reference elevation
        temp_terminus = \
            vascaling.compute_temp_terminus(ref_t, temp_grad, ref_hgt=ref_h,
                                            terminus_hgt=ref_h,
                                            temp_anomaly=temp_anomaly)
        np.testing.assert_allclose(temp_terminus, ref_t + temp_anomaly)

        # the terminus temperature must equal the input temperature
        # if the gradient is zero
        for term_h in np.array([-100, 0, 100]) + ref_h:
            temp_terminus = \
                vascaling.compute_temp_terminus(ref_t, temp_grad=0,
                                                ref_hgt=ref_h,
                                                terminus_hgt=term_h,
                                                temp_anomaly=temp_anomaly)
            np.testing.assert_allclose(temp_terminus, ref_t + temp_anomaly)

        # now test the routine with actual elevation differences
        # and a non zero temperature gradient
        for h_diff in np.array([-100, 0, 100]):
            term_h = ref_h + h_diff
            temp_diff = temp_grad * h_diff
            temp_terminus = \
                vascaling.compute_temp_terminus(ref_t, temp_grad,
                                                ref_hgt=ref_h,
                                                terminus_hgt=term_h,
                                                temp_anomaly=temp_anomaly)
            np.testing.assert_allclose(temp_terminus,
                                       ref_t + temp_anomaly + temp_diff)

    def test_solid_prcp(self):
        """Tests the subroutine which computes solid precipitation amount from
        given total precipitation and temperature.
        """
        with utils.ncDataset(get_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]

        temp_all_solid = 0
        temp_grad = -0.0065
        min_hgt = ref_h - 100
        max_hgt = ref_h + 100

        # if the terminus temperature is below the threshold for
        # solid precipitation all fallen precipitation must be solid
        temp_terminus = ref_t * 0 + temp_all_solid
        solid_prcp = vascaling.compute_solid_prcp(ref_p, min_hgt, max_hgt,
                                                  temp_terminus,
                                                  temp_all_solid, temp_grad)
        np.testing.assert_allclose(solid_prcp, ref_p)

        # if the temperature at the maximal elevation is above the threshold
        # for solid precipitation all fallen precipitation must be liquid
        temp_terminus = ref_t + 100
        solid_prcp = vascaling.compute_solid_prcp(ref_p, min_hgt, max_hgt,
                                                  temp_terminus,
                                                  temp_all_solid, temp_grad)
        np.testing.assert_allclose(solid_prcp, 0)

        # test extreme case if max_hgt equals min_hgt
        test_p = ref_p * (ref_t <= temp_all_solid).astype(int)
        solid_prcp = vascaling.compute_solid_prcp(ref_p, ref_h, ref_h, ref_t,
                                                  temp_all_solid, temp_grad)
        np.testing.assert_allclose(solid_prcp, test_p)

    def test_min_max_elevation(self):
        """Test the helper method which computes the minimal and maximal
        glacier surface elevation.
        """
        gdir = self._gdir(flowlines=False)
        min_hgt, max_hgt = vascaling.get_min_max_elevation(gdir)
        np.testing.assert_allclose(min_hgt, 2430, rtol=1e-2)
        np.testing.assert_allclose(max_hgt, 3674, rtol=1e-2)

    # -- mass balance model

    def test_monthly_climate(self):
        """The sum over the monthly climate must equal the annual climate."""
        gdir = self._calibrated_gdir()
        mbmod = vascaling.VAScalingMassBalance(gdir)

        year = 1975
        _, temp_annual, _, prcp_annual = mbmod.get_annual_climate(None,
                                                                  year=year)

        temp_months = 0.
        prcp_months = 0.
        for month in np.arange(1, 13):
            yr = utils.date_to_floatyear(year, month)
            _, t, _, p = mbmod.get_monthly_climate(None, year=yr)
            temp_months += t
            prcp_months += p

        np.testing.assert_allclose(temp_annual, temp_months, rtol=1e-6)
        np.testing.assert_allclose(prcp_annual, prcp_months, rtol=1e-6)

    def test_annual_mb(self):
        """Test the routine computing the annual mass balance against the
        mass balance equation evaluated by hand.
        """
        gdir = self._calibrated_gdir()
        mbmod = vascaling.VAScalingMassBalance(gdir)

        year = 1975
        _, temp, _, prcp = mbmod.get_annual_climate(None, year=year)

        # specify scaling factor for SI units [kg s-1]
        fac_SI = cfg.SEC_IN_YEAR * cfg.PARAMS['ice_density']

        # compute mass balance 'by hand' and compare
        mb_ref = (prcp - mbmod.melt_f * temp) / fac_SI
        mb_mod = mbmod.get_annual_mb(None, year=year)
        np.testing.assert_allclose(mb_ref, mb_mod, rtol=1e-6)

        # now with a bias
        bias = 100.
        mbmod = vascaling.VAScalingMassBalance(gdir, bias=bias)
        mb_ref = (prcp - mbmod.melt_f * temp - bias) / fac_SI
        mb_mod = mbmod.get_annual_mb(None, year=year)
        np.testing.assert_allclose(mb_ref, mb_mod, rtol=1e-6)

    def test_monthly_specific_mb(self):
        """The monthly specific mass balances must sum up to the annual one."""
        gdir = self._calibrated_gdir()
        mbmod = vascaling.VAScalingMassBalance(gdir)

        year = 1975
        mb_annual = mbmod.get_specific_mb(year=year)
        months = [utils.date_to_floatyear(year, m) for m in np.arange(1, 13)]
        mb_monthly = mbmod.get_specific_mb(year=months,
                                           time_resolution='monthly')
        np.testing.assert_allclose(mb_annual, np.sum(mb_monthly), rtol=1e-6)

    def test_specific_mb_vs_oggm(self):
        """The VAS and the OGGM mass balance model must tell a similar story
        when calibrated on the same reference mass balance.
        """
        gdir = self._calibrated_gdir()
        years = np.arange(*REF_MB_YEARS)
        mb_vas = vascaling.VAScalingMassBalance(gdir).get_specific_mb(
            year=years)

        # now calibrate OGGM's own model on the same value
        massbalance.mb_calibration_from_scalar_mb(
            gdir, ref_mb=REF_MB, ref_mb_years=REF_MB_YEARS,
            calibrate_param1='melt_f', calibrate_param2='prcp_fac',
            calibrate_param3='temp_bias', overwrite_gdir=True,
            overwrite_observations=True)
        fls = gdir.read_pickle('inversion_flowlines')
        mb_oggm = massbalance.MonthlyTIModel(gdir).get_specific_mb(
            fls=fls, year=years)

        # both must reproduce the reference mass balance
        np.testing.assert_allclose(mb_vas.mean(), REF_MB, atol=1e-3)
        np.testing.assert_allclose(mb_oggm.mean(), REF_MB, atol=1e-3)
        # and the interannual variability must agree
        assert corrcoef(mb_vas, mb_oggm) >= 0.9

    # -- calibration

    def test_mb_calibration(self):
        """The VAS mass balance model must plug into OGGM's calibration."""
        gdir = self._gdir()

        df = massbalance.mb_calibration_from_scalar_mb(
            gdir, ref_mb=REF_MB, ref_mb_years=REF_MB_YEARS,
            mb_model_class=vascaling.VAScalingMassBalance,
            calibrate_param1='melt_f', calibrate_param2='prcp_fac',
            calibrate_param3='temp_bias', overwrite_gdir=True,
            overwrite_observations=True)

        # the calibrated parameters are written to the glacier settings
        for key in ['melt_f', 'prcp_fac', 'temp_bias']:
            assert gdir.settings[key] == df[key]
        assert cfg.PARAMS['melt_f_min'] < df['melt_f'] < cfg.PARAMS['melt_f_max']
        assert df['bias'] == 0

        # and the calibrated model reproduces the reference mass balance
        mbmod = vascaling.VAScalingMassBalance(gdir)
        mb = mbmod.get_specific_mb(year=np.arange(*REF_MB_YEARS))
        np.testing.assert_allclose(mb.mean(), REF_MB, atol=1e-3)

    def test_mb_calibration_geodetic_wrapper(self):
        """Our wrapper must give the same answer as OGGM's task."""
        gdir = self._calibrated_gdir()
        expected = {k: gdir.settings[k]
                    for k in ['melt_f', 'prcp_fac', 'temp_bias']}

        df = vascaling.mb_calibration_from_geodetic_mb(
            gdir, use_observations_file=True, overwrite_gdir=True)
        for key, value in expected.items():
            np.testing.assert_allclose(df[key], value)

    def test_multiple_flowlines_refused(self):
        """The VAS model describes the glacier as a whole."""
        gdir = self._calibrated_gdir()
        mbmod = vascaling.VAScalingMassBalance(gdir)
        fls = gdir.read_pickle('inversion_flowlines')
        with pytest.raises(oggm.exceptions.InvalidWorkflowError):
            mbmod.get_specific_mb(fls=fls * 2, year=1975)

    def test_prcp_clim(self):
        """The turnover must be the mean solid precipitation over the whole
        climate record, and must not depend on the terminus position.
        """
        gdir = self._calibrated_gdir()
        mbmod = vascaling.VAScalingMassBalance(gdir)

        years = np.unique(mbmod.years)
        prcp_sol = np.array([mbmod.get_annual_climate(None, year=y)[3].sum()
                             for y in years])
        np.testing.assert_allclose(mbmod.prcp_clim, prcp_sol.mean() * 1e-3)

        # the turnover is a climatology at the RGI date geometry
        mbmod.min_hgt = mbmod.min_hgt_0 + 300
        np.testing.assert_allclose(mbmod.prcp_clim, prcp_sol.mean() * 1e-3)

        # it must not depend on anything the calibration writes, otherwise
        # a glacier evolves differently during and after a calibration
        sub = vascaling.VAScalingMassBalance(
            gdir, prcp_clim_period='1953-01-01_2003-01-01')
        assert sub.prcp_clim != mbmod.prcp_clim

    # -- dynamical model

    def test_time_scales(self):
        """Test the internal method which computes the glaciers time scales
        for length change and area change.
        """
        _, model = self._vas_model()
        model._compute_time_scales()
        # the response times moved when prcp_clim stopped being defined
        # around t*, but they must stay in the same ballpark
        assert 20 < model.tau_l < 60
        assert 5 < model.tau_a < 25
        assert model.tau_l > model.tau_a

    def test_reset(self):
        """Test the method which sets the model back to its initial state."""
        _, model = self._vas_model()
        model.run_until(model.year + 10)
        model.reset()
        assert model.year == model.year_0
        assert model.length_m == model.length_m_0
        assert model.area_m2 == model.area_m2_0
        assert model.volume_m3 == model.volume_m3_0
        assert model.min_hgt == model.min_hgt_0

    def test_step(self):
        """Test the advance of the model glacier after one time step."""
        _, model = self._vas_model()
        m0 = copy.deepcopy(model)
        model.step()
        dV = m0.spec_mb * m0.area_m2 / m0.rho
        np.testing.assert_allclose(model.volume_m3 - m0.volume_m3, dV)

    def test_ice_cap(self):
        """Ice caps use their own scaling parameters, and must keep them
        through `create_start_glacier`.
        """
        gdir, model = self._vas_model()
        cap = vascaling.VAScalingModel(
            year_0=model.year_0, area_m2_0=model.area_m2_0,
            min_hgt=model.min_hgt_0, max_hgt=model.max_hgt,
            mb_model=model.mb_model, glacier_type='Ice cap')
        assert cap.gamma == cfg.PARAMS['vas_gamma_icecap_area']
        cap.create_start_glacier(model.area_m2_0 / 2, year_start=model.year_0)
        assert cap.glacier_type == 'Ice cap'
        assert cap.gamma == cfg.PARAMS['vas_gamma_icecap_area']

        with pytest.raises(oggm.exceptions.InvalidParamsError):
            vascaling.VAScalingModel(
                year_0=model.year_0, area_m2_0=model.area_m2_0,
                min_hgt=model.min_hgt_0, max_hgt=model.max_hgt,
                mb_model=model.mb_model, glacier_type='Snowball')

    def test_run_until_and_store(self):
        """The stored diagnostics must match the model state, and be readable
        by OGGM's `compile_run_output`.
        """
        gdir, model = self._vas_model(y0=1950)
        diag_path = gdir.get_filepath('model_diagnostics')
        vas_diag_path = gdir.get_filepath('vas_diagnostics')
        ds = model.run_until_and_store(2000, diag_path=diag_path,
                                       vas_diag_path=vas_diag_path)

        # calendar years, as OGGM does it
        np.testing.assert_array_equal(ds.time, np.arange(1950, 2001))
        np.testing.assert_array_equal(ds.calendar_year, np.arange(1950, 2001))
        # the last entry must be the current model state
        np.testing.assert_allclose(ds.volume_m3[-1], model.volume_m3)
        np.testing.assert_allclose(ds.area_m2[-1], model.area_m2)
        np.testing.assert_allclose(ds.length_m[-1], model.length_m)
        np.testing.assert_allclose(ds.min_hgt[-1], model.min_hgt)

        # the OGGM compatible file holds the geometry only
        out = utils.compile_run_output([gdir], path=False)
        np.testing.assert_allclose(out.volume.values.flatten(),
                                   ds.volume_m3.values)
        np.testing.assert_allclose(out.area.values.flatten(),
                                   ds.area_m2.values)

    def test_run_from_climate_data(self):
        """The entity task must run and write both diagnostics files."""
        gdir = self._calibrated_gdir()
        model = vascaling.run_from_climate_data(gdir, ys=1950, ye=2000)
        assert gdir.has_file('model_diagnostics')
        assert gdir.has_file('vas_diagnostics')
        assert model.year == 2000
        # Hintereisferner lost mass over this period
        assert model.volume_m3 < model.volume_m3_0

    def test_run_random_climate(self):
        """A random climate run around a year with a balanced mass budget
        must keep the glacier roughly in place.
        """
        gdir = self._calibrated_gdir()
        # find a year around which the glacier is close to equilibrium
        _, model = self._vas_model(gdir=gdir)
        model.mb_model.min_hgt = model.min_hgt_0

        vascaling.run_random_climate(gdir, nyears=300, y0=1930, seed=1,
                                     halfsize=15, output_filesuffix='_rdn')
        with xr.open_dataset(gdir.get_filepath('vas_diagnostics',
                                               filesuffix='_rdn')) as ds:
            ds = ds.load()
        # the volume must stay finite and positive, and vary
        assert np.all(np.isfinite(ds.volume_m3))
        assert ds.volume_m3.min() > 0
        assert ds.volume_m3.std() > 0

        # a colder climate must give a bigger glacier, a warmer one a smaller
        vascaling.run_random_climate(gdir, nyears=300, y0=1930, seed=1,
                                     halfsize=15, temperature_bias=-0.5,
                                     output_filesuffix='_cold')
        vascaling.run_random_climate(gdir, nyears=300, y0=1930, seed=1,
                                     halfsize=15, temperature_bias=+0.5,
                                     output_filesuffix='_warm')
        with xr.open_dataset(gdir.get_filepath('vas_diagnostics',
                                               filesuffix='_cold')) as dsc:
            v_cold = float(dsc.volume_m3[-1])
        with xr.open_dataset(gdir.get_filepath('vas_diagnostics',
                                               filesuffix='_warm')) as dsw:
            v_warm = float(dsw.volume_m3[-1])
        assert v_cold > float(ds.volume_m3[-1]) > v_warm

    def test_run_constant_climate(self):
        """A constant climate run must converge towards an equilibrium."""
        gdir = self._calibrated_gdir()
        vascaling.run_constant_climate(gdir, nyears=600, y0=1930,
                                       halfsize=15, output_filesuffix='_cst')
        with xr.open_dataset(gdir.get_filepath('vas_diagnostics',
                                               filesuffix='_cst')) as ds:
            ds = ds.load()
        # the specific mass balance must approach zero
        assert abs(float(ds.spec_mb[-1])) < 10
        # and the volume must stop changing
        v = ds.volume_m3.values
        rate = abs(v[-1] - v[-50]) / v[-1]
        assert rate < 1e-3

    def test_run_until_equilibrium(self):
        """Test the equilibrium search."""
        gdir = self._calibrated_gdir()
        mbmod = vascaling.ConstantVASMassBalance(gdir, y0=1930, halfsize=15)
        min_hgt, max_hgt = vascaling.get_min_max_elevation(gdir)
        model = vascaling.VAScalingModel(year_0=0,
                                         area_m2_0=gdir.rgi_area_m2,
                                         min_hgt=min_hgt, max_hgt=max_hgt,
                                         mb_model=mbmod,
                                         glacier_type=gdir.glacier_type)
        model.run_until_equilibrium(rate=1e-4)
        assert abs(model.spec_mb) < 10

        # it only works with a constant climate
        _, model = self._vas_model(gdir=gdir)
        with pytest.raises(TypeError):
            model.run_until_equilibrium()

    def test_find_start_area(self):
        """The start area must actually reproduce the RGI area."""
        gdir = self._calibrated_gdir()
        area_start = vascaling.find_start_area(gdir, year_start=1851)

        # re-run from that area and check we land on the RGI area
        mbmod = vascaling.VAScalingMassBalance(gdir)
        target_yr = vascaling.core._target_year(gdir, mbmod, None)
        model_ref = vascaling.core._reference_model(gdir, mbmod, target_yr)
        model = vascaling.core._start_model(model_ref, area_start, 1851)
        model.run_until(target_yr)
        np.testing.assert_allclose(model.area_m2, gdir.rgi_area_m2, rtol=1e-4)

    def test_find_start_area_raises_instead_of_hitting_the_bound(self):
        """A start area that cannot be reached must raise, not be returned.

        `minimize_scalar(method='bounded')` converges onto its own bound and
        reports success, which used to hand back a silent non-match.
        """
        gdir = self._calibrated_gdir()
        with pytest.raises(RuntimeError) as err:
            vascaling.find_start_area(gdir, year_start=1851,
                                      max_area_factor=1.0)
        assert 'reproduces' in str(err.value)

    def test_run_reconstruction(self):
        """The reconstruction must pass through the observed area."""
        gdir = self._calibrated_gdir()
        model = vascaling.run_reconstruction(gdir, ys=1950,
                                             output_filesuffix='_rec')

        mbmod = vascaling.VAScalingMassBalance(gdir)
        target_yr = vascaling.core._target_year(gdir, mbmod, None)
        with xr.open_dataset(gdir.get_filepath('vas_diagnostics',
                                               filesuffix='_rec')) as ds:
            ds = ds.load()

        # the run goes from the requested start to the end of the record
        assert int(ds.time[0]) == 1950
        assert int(ds.time[-1]) == int(mbmod.ye) + 1
        # and the RGI area is matched at the inventory date
        np.testing.assert_allclose(float(ds.area_m2.sel(time=target_yr)),
                                   gdir.rgi_area_m2, rtol=1e-4)
        # which `run_from_climate_data` would not do from the same start year
        naive = vascaling.run_from_climate_data(gdir, ys=1950,
                                                ye=target_yr,
                                                output_filesuffix='_naive')
        assert (abs(naive.area_m2 - gdir.rgi_area_m2) >
                abs(float(ds.area_m2.sel(time=target_yr)) - gdir.rgi_area_m2))
        assert model.year == int(mbmod.ye) + 1

    def test_mb_calibration_dynamic(self):
        """The dynamic calibration must match the mass change of the
        *evolving* glacier, not of the fixed RGI geometry.
        """
        gdir = self._gdir()
        period = '1953-01-01_2003-01-01'
        # a 1950 start keeps the terminus inside the DEM range, see the
        # note on the terminus elevation parameterisation in the docstring
        df = vascaling.mb_calibration_dynamic_from_geodetic_mb(
            gdir, ref_mb=REF_MB, ref_mb_period=period, prcp_fac=2.5,
            ys=1950, overwrite_gdir=True)

        assert cfg.PARAMS['melt_f_min'] <= df['melt_f'] <= cfg.PARAMS['melt_f_max']
        assert df['vas_dynamic_calibration'] is True
        # it must have matched the observation
        np.testing.assert_allclose(df['vas_dmdtda_mismatch'], 0, atol=1.)

        # and the reconstruction must reproduce that mass change
        model = vascaling.run_reconstruction(gdir, ys=1950,
                                             output_filesuffix='_dyn')
        with xr.open_dataset(gdir.get_filepath('vas_diagnostics',
                                               filesuffix='_dyn')) as ds:
            ds = ds.load()
        dmdtda = ((float(ds.volume_m3.sel(time=2003)) -
                   float(ds.volume_m3.sel(time=1953))) *
                  cfg.PARAMS['ice_density'] / gdir.rgi_area_m2 / 50)
        np.testing.assert_allclose(dmdtda, REF_MB, atol=1.)

    def test_fixed_geometry_mass_balance(self):
        """The fixed geometry series must match the mass balance model."""
        gdir = self._calibrated_gdir()
        odf = vascaling.fixed_geometry_mass_balance(gdir, ys=1950, ye=2000)
        mbmod = vascaling.VAScalingMassBalance(gdir)
        ref = mbmod.get_specific_mb(year=np.arange(1950, 2001))
        np.testing.assert_allclose(odf.values, ref)

    def test_file_model(self):
        """The FileModel must reproduce the stored run."""
        gdir = self._calibrated_gdir()
        vascaling.run_from_climate_data(gdir, ys=1950, ye=2000,
                                        output_filesuffix='_hist')
        fp = gdir.get_filepath('vas_diagnostics', filesuffix='_hist')
        fmod = vascaling.FileModel(fp)
        with xr.open_dataset(fp) as ds:
            ds = ds.load()
        fmod.run_until(1975)
        np.testing.assert_allclose(fmod.area_m2,
                                   float(ds.area_m2.sel(time=1975)))
        np.testing.assert_allclose(fmod.min_hgt,
                                   float(ds.min_hgt.sel(time=1975)))
        fmod.reset()
        np.testing.assert_allclose(fmod.area_m2, fmod.area_m2_0)
