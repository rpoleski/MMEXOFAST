# tests/test_emcee_fit_results.py
"""
Unit tests for the refactored emcee fitting pipeline.

Changes from the old test file
-------------------------------
- EmceeFitResults (results.py) is removed; format_results_as_df() moves
  into MMEXOFASTFitResults.
- EmceeFitResults (fitters.py) replaces MinimalResults, holding the full
  sampler and lazily computing percentiles/sigmas.
- EmceeLCFitter.run() now sets self.results to an EmceeFitResults instance,
  consistent with SFitFitter.results.

Test class mapping from old to new
------------------------------------
Old TestWidePlanetFitterBestTheta      → TestEmceeLCFitterResultsAfterRun (added)
Old TestEmceeFitResults (results.py)   → TestMMEXOFASTFitResultsWithEmcee (moved)
Old TestEmceeFitResultsWithFixedParams → TestMMEXOFASTFitResultsWithEmceeFixedParams (moved)
Old TestFitRecordWithEmcee             → TestFitRecordWithEmcee (merged)

Duplicated tests dropped
-------------------------
test_get_params_excludes_chi2, test_get_sigmas_contains_all_fitted_params,
test_format_has_required_columns, test_format_returns_dataframe,
test_to_dataframe_has_sigma_*_column (merged), test_from_full_result_is_complete
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import emcee

from mmexofast.fitters import EmceeFitResults, EmceeLCFitter
from mmexofast.results import MMEXOFASTFitResults, FitRecord
from mmexofast.fit_types import (
    FitKey, LensType, SourceType, ParallaxBranch, LensOrbMotion
)

# ===========================================================================
# Module-level constants
# ===========================================================================

_DEFAULT_PARAMETERS_TO_FIT = [
    't_0', 'u_0', 't_E', 'log_rho', 'log_s', 'log_q', 'alpha'
]
_TRUE_VALUES_MAP = {
    't_0': 2459000.0, 'u_0': 0.1, 't_E': 20.0,
    'log_rho': -2.0,  'log_s': 0.1, 'log_q': -3.0, 'alpha': 270.0,
}
_STEP_SIZES_MAP = {
    't_0': 0.1,    'u_0': 0.001, 't_E': 0.5,
    'log_rho': 0.01, 'log_s': 0.01, 'log_q': 0.01, 'alpha': 1.0,
}

_N_WALKERS = 20
_N_STEPS   = 100
_N_BURN    = 20

_N_GOOD_DATASET_1 = 100
_N_GOOD_DATASET_2 = 80
_N_DATA = _N_GOOD_DATASET_1 + _N_GOOD_DATASET_2   # 180

_SOURCE_FLUXES = [np.array([1000.0]), np.array([500.0])]
_BLEND_FLUXES  = [100.0, 50.0]

_MOCK_MAG_I_SOURCE_OGLE = 22 - 2.5 * np.log10(float(np.squeeze(_SOURCE_FLUXES[0])))
_MOCK_MAG_I_BLEND_OGLE  = 22 - 2.5 * np.log10(float(np.squeeze(_BLEND_FLUXES[0])))
_MOCK_MAG_R_SOURCE_MOA  = 22 - 2.5 * np.log10(float(np.squeeze(_SOURCE_FLUXES[1])))
_MOCK_MAG_R_BLEND_MOA   = 22 - 2.5 * np.log10(float(np.squeeze(_BLEND_FLUXES[1])))

_EXPECTED_FLUX_PARAM_NAMES = ['I_S_OGLE', 'I_B_OGLE', 'R_S_MOA', 'R_B_MOA']
_EXPECTED_FLUX_PARAM_VALUES = {
    'I_S_OGLE': _MOCK_MAG_I_SOURCE_OGLE,
    'I_B_OGLE': _MOCK_MAG_I_BLEND_OGLE,
    'R_S_MOA':  _MOCK_MAG_R_SOURCE_MOA,
    'R_B_MOA':  _MOCK_MAG_R_BLEND_MOA,
}

_MAG_FROM_FLUX_PATCH_PATH = 'MulensModel.utils.Utils.get_mag_and_err_from_flux'


# ===========================================================================
# Module-level helpers
# ===========================================================================

def _get_parameter_name(param):
    """Strip log_ prefix. Mirrors EmceeLCFitter.get_parameter_name."""
    if param.startswith('log_'):
        return param[4:]
    return param


def _mock_get_mag_and_err_from_flux(flux, err_flux):
    """Stand-in using mag = 22 - 2.5*log10(flux)."""
    return 22 - 2.5 * np.log10(flux), 0.0


def make_mock_sampler(n_walkers=10, n_steps=300, n_dim=3, seed=42):
    """Return a mock EnsembleSampler with a reproducible Gaussian chain."""
    rng = np.random.default_rng(seed)
    chain = rng.standard_normal((n_walkers, n_steps, n_dim))
    lnprobability = -rng.standard_exponential((n_walkers, n_steps))
    sampler = MagicMock(spec=emcee.EnsembleSampler)
    sampler.chain               = chain
    sampler.lnprobability       = lnprobability
    sampler.iteration           = n_steps
    sampler.acceptance_fraction = np.full(n_walkers, 0.3)
    sampler.sample.return_value = iter([])
    return sampler


def make_mock_dataset(label='OGLE-I', n_good=100):
    """Return a minimal mock MulensData dataset."""
    dataset = MagicMock()
    dataset.good = np.ones(n_good, dtype=bool)
    dataset.plot_properties = {'label': label}
    dataset.bandpass = 'I'
    return dataset


def make_mock_emcee_fitter(
        parameters_to_fit=None,
        fixed_params_dict=None,
        seed=42):
    """
    Build a mock EmceeLCFitter in a post-run state.

    Updated from the old helper: mock_fitter.results is now set to a real
    EmceeFitResults instance (fitters.py), and emcee_settings is expanded
    to include all keys.

    Returns
    -------
    mock_fitter : MagicMock
        Post-run fitter with .results, .best, .best_theta, .sampler, etc.
    parameters_to_fit : list of str
    best_dict : dict
        Max-likelihood params in linear space + 'chi2'.
    best_theta : np.ndarray, shape (n_params,)
        Max-likelihood emcee vector.
    expected : dict
        Keys: 'p16', 'p50', 'p84', 'sigma_minus', 'sigma_plus',
        all shape (n_params,), derived independently from post-burn-in chain.
    """
    np.random.seed(seed)

    if parameters_to_fit is None:
        parameters_to_fit = list(_DEFAULT_PARAMETERS_TO_FIT)

    n_params    = len(parameters_to_fit)
    true_values = np.array([_TRUE_VALUES_MAP[p] for p in parameters_to_fit])
    step_sizes  = np.array([_STEP_SIZES_MAP[p]  for p in parameters_to_fit])

    chain = true_values + (
        np.random.randn(_N_WALKERS, _N_STEPS, n_params) * step_sizes
    )
    lnprobability = -0.5 * (
        _N_DATA + np.sum(((chain - true_values) / step_sizes) ** 2, axis=2)
    )

    samples    = chain[:, _N_BURN:, :].reshape((-1, n_params))
    prob       = lnprobability[:, _N_BURN:].reshape(-1)
    best_theta = samples[np.argmax(prob)]

    chi2_best = _N_DATA + np.sum(((best_theta - true_values) / step_sizes) ** 2)

    best_dict = {}
    for i, param in enumerate(parameters_to_fit):
        linear_key = _get_parameter_name(param)
        best_dict[linear_key] = (
            10. ** best_theta[i] if param.startswith('log_') else best_theta[i]
        )
    if fixed_params_dict is not None:
        best_dict.update(fixed_params_dict)
    best_dict['chi2'] = chi2_best

    p = np.percentile(samples, [16, 50, 84], axis=0)
    expected = {
        'p16':         p[0],
        'p50':         p[1],
        'p84':         p[2],
        'sigma_minus': p[1] - p[0],
        'sigma_plus':  p[2] - p[1],
    }

    mock_sampler = MagicMock()
    mock_sampler.chain         = chain
    mock_sampler.lnprobability = lnprobability

    mock_dataset_1 = MagicMock()
    mock_dataset_1.plot_properties = {'label': 'n20100309.I.OGLE.OB08092.txt'}
    mock_dataset_1.good = np.ones(_N_GOOD_DATASET_1, dtype=bool)

    mock_dataset_2 = MagicMock()
    mock_dataset_2.plot_properties = {'label': 'n20100309.R.MOA.OB08092.txt'}
    mock_dataset_2.good = np.ones(_N_GOOD_DATASET_2, dtype=bool)

    mock_event = MagicMock()
    mock_event.source_fluxes = list(_SOURCE_FLUXES)
    mock_event.blend_fluxes  = list(_BLEND_FLUXES)

    emcee_settings = {
        'n_burn':    _N_BURN,
        'n_dim':     n_params,
        'n_walkers': _N_WALKERS,
        'n_steps':   _N_STEPS,
        'acceptance_fraction': 0.1,
    }

    mock_fitter = MagicMock()
    mock_fitter.parameters_to_fit  = parameters_to_fit
    mock_fitter.best               = best_dict
    mock_fitter.best_theta         = best_theta
    mock_fitter.sampler            = mock_sampler
    mock_fitter.emcee_settings     = emcee_settings
    mock_fitter.get_parameter_name = MagicMock(side_effect=_get_parameter_name)
    mock_fitter.datasets           = [mock_dataset_1, mock_dataset_2]
    mock_fitter.get_best_fit_event.return_value = mock_event
    # NEW: .results is a real EmceeFitResults instance, not None
    mock_fitter.results = EmceeFitResults(
        sampler=mock_sampler,
        x=best_theta,
        emcee_settings=emcee_settings,
        parameters_to_fit=parameters_to_fit,
    )

    return mock_fitter, parameters_to_fit, best_dict, best_theta, expected


# ===========================================================================
# Module-level fixtures  (Groups 1 and 2 only)
# ===========================================================================

@pytest.fixture
def n_walkers():
    return 10

@pytest.fixture
def n_steps():
    return 300

@pytest.fixture
def n_dim():
    return 3

@pytest.fixture
def n_burn():
    return 100

@pytest.fixture
def emcee_settings(n_walkers, n_steps, n_dim, n_burn):
    return {
        'n_walkers': n_walkers, 'n_burn': n_burn,
        'n_steps': n_steps, 'n_dim': n_dim,
        'acceptance_fraction': 0.1,
    }

@pytest.fixture
def parameters_to_fit():
    return ['t_0', 'u_0', 't_E']

@pytest.fixture
def best_theta():
    return np.array([2460000.0, 0.5, 20.0])

@pytest.fixture
def mock_sampler(n_walkers, n_steps, n_dim):
    return make_mock_sampler(n_walkers=n_walkers, n_steps=n_steps, n_dim=n_dim)

@pytest.fixture
def emcee_fit_results(mock_sampler, best_theta, emcee_settings, parameters_to_fit):
    return EmceeFitResults(
        sampler=mock_sampler,
        x=best_theta,
        emcee_settings=emcee_settings,
        parameters_to_fit=parameters_to_fit,
    )


# ===========================================================================
# Group 1: EmceeFitResults (fitters.py, renamed from MinimalResults)
# ===========================================================================

class TestEmceeFitResults:
    """
    Tests for EmceeFitResults (fitters.py).

    Verifies stored attributes, lazy percentile computation from the
    post-burn-in chain, derived uncertainties, and that x is best_theta
    (max-likelihood), not p50 (median).
    """

    # --- Stored attributes --------------------------------------------------

    def test_stores_sampler(self, emcee_fit_results, mock_sampler):
        assert emcee_fit_results.sampler is mock_sampler

    def test_stores_x_as_best_theta(self, emcee_fit_results, best_theta):
        np.testing.assert_array_equal(emcee_fit_results.x, best_theta)

    def test_stores_parameters_to_fit(self, emcee_fit_results, parameters_to_fit):
        assert emcee_fit_results.parameters_to_fit == parameters_to_fit

    def test_stores_emcee_settings(self, emcee_fit_results, emcee_settings):
        assert emcee_fit_results.emcee_settings == emcee_settings

    # --- Percentiles --------------------------------------------------------

    def test_percentiles_shape(self, emcee_fit_results, n_dim):
        """Shape is (3, n_dim): rows are p16, p50, p84."""
        assert emcee_fit_results.percentiles.shape == (3, n_dim)

    def test_percentiles_use_post_burnin_chain_only(
            self, emcee_fit_results, mock_sampler, emcee_settings):
        n_burn   = emcee_settings['n_burn']
        n_dim    = emcee_settings['n_dim']
        samples  = mock_sampler.chain[:, n_burn:, :].reshape((-1, n_dim))
        expected = np.percentile(samples, [16, 50, 84], axis=0)
        np.testing.assert_array_almost_equal(emcee_fit_results.percentiles, expected)

    def test_percentiles_are_cached(self, emcee_fit_results):
        p1 = emcee_fit_results.percentiles
        p2 = emcee_fit_results.percentiles
        assert p1 is p2

    # --- Uncertainties ------------------------------------------------------

    def test_sigma_minus_is_p50_minus_p16(self, emcee_fit_results):
        p = emcee_fit_results.percentiles
        np.testing.assert_array_almost_equal(emcee_fit_results.sigma_minus, p[1] - p[0])

    def test_sigma_plus_is_p84_minus_p50(self, emcee_fit_results):
        p = emcee_fit_results.percentiles
        np.testing.assert_array_almost_equal(emcee_fit_results.sigma_plus, p[2] - p[1])

    def test_sigmas_is_mean_of_asymmetric(self, emcee_fit_results):
        expected = (emcee_fit_results.sigma_minus + emcee_fit_results.sigma_plus) / 2
        np.testing.assert_array_almost_equal(emcee_fit_results.sigmas, expected)

    def test_sigmas_are_positive(self, emcee_fit_results):
        assert np.all(emcee_fit_results.sigmas > 0)

    # --- x is best_theta, not p50 -------------------------------------------

    def test_x_is_best_theta_not_p50(
            self, mock_sampler, emcee_settings, parameters_to_fit):
        """x is the max-likelihood value supplied at construction, not p50."""
        x_distinct = np.array([99.0, 99.0, 99.0])  # well outside the chain
        results = EmceeFitResults(
            sampler=mock_sampler,
            x=x_distinct,
            emcee_settings=emcee_settings,
            parameters_to_fit=parameters_to_fit,
        )
        np.testing.assert_array_equal(results.x, x_distinct)
        assert not np.allclose(results.x, results.percentiles[1])


# ===========================================================================
# Group 2: EmceeLCFitter.results contract after run()
# ===========================================================================

class TestEmceeLCFitterResultsAfterRun:
    """
    After run(), EmceeLCFitter.results must be a populated EmceeFitResults
    instance, consistent with SFitFitter.results being the raw sfit object.

    best_theta contract tests are incorporated from old TestWidePlanetFitterBestTheta.
    The log-space round-trip tests from that class are dropped: they tested
    mock consistency rather than fitter behavior and cannot be verified
    when best is read from a mock event.
    """

    @pytest.fixture
    def fitter_before_run(self):
        return EmceeLCFitter(
            datasets=[make_mock_dataset()],
            initial_guess={'t_0': 2460000.0, 'u_0': 0.5, 't_E': 20.0},
            emcee_settings={
                'n_walkers': 4, 'n_burn': 2, 'n_steps': 5,
                'acceptance_fraction': None,
            },
        )

    @pytest.fixture
    def fitter_after_run(self, fitter_before_run):
        fitter       = fitter_before_run
        mock_sampler = make_mock_sampler(n_walkers=4, n_steps=5, n_dim=3)

        mock_event = MagicMock()
        mock_event.model.parameters.parameters = {
            't_0': 2460000.0, 'u_0': 0.5, 't_E': 20.0
        }
        mock_event.get_chi2.return_value = 100.0

        def fake_initialize():
            fitter._event = mock_event

        with patch.object(fitter, 'initialize_event', side_effect=fake_initialize), \
             patch('emcee.EnsembleSampler', return_value=mock_sampler):
            fitter.run()

        return fitter

    # --- results contract ---------------------------------------------------

    def test_results_is_none_before_run(self, fitter_before_run):
        assert fitter_before_run.results is None

    def test_results_is_emcee_fit_results_after_run(self, fitter_after_run):
        assert isinstance(fitter_after_run.results, EmceeFitResults)

    def test_results_x_equals_best_theta(self, fitter_after_run):
        np.testing.assert_array_equal(
            fitter_after_run.results.x, fitter_after_run.best_theta)

    def test_results_sampler_is_fitter_sampler(self, fitter_after_run):
        assert fitter_after_run.results.sampler is fitter_after_run.sampler

    def test_results_parameters_to_fit_match(self, fitter_after_run):
        assert (fitter_after_run.results.parameters_to_fit
                == fitter_after_run.parameters_to_fit)

    def test_results_emcee_settings_match(self, fitter_after_run):
        assert (fitter_after_run.results.emcee_settings
                == fitter_after_run.emcee_settings)

    # --- best_theta contract (from old TestWidePlanetFitterBestTheta) -------

    def test_best_theta_is_numpy_array(self, fitter_after_run):
        assert isinstance(fitter_after_run.best_theta, np.ndarray)

    def test_best_theta_has_correct_shape(self, fitter_after_run):
        n_params = len(fitter_after_run.parameters_to_fit)
        assert fitter_after_run.best_theta.shape == (n_params,)

    def test_best_theta_is_max_likelihood_post_burnin_sample(self, fitter_after_run):
        """best_theta must equal the post-burn-in sample with highest lnprobability."""
        n_burn  = fitter_after_run.emcee_settings['n_burn']
        n_dim   = fitter_after_run.emcee_settings['n_dim']
        samples = fitter_after_run.sampler.chain[:, n_burn:, :].reshape((-1, n_dim))
        prob    = fitter_after_run.sampler.lnprobability[:, n_burn:].reshape(-1)
        np.testing.assert_array_equal(
            fitter_after_run.best_theta, samples[np.argmax(prob)])


# ===========================================================================
# Group 3: MMEXOFASTFitResults wrapping EmceeLCFitter
# ===========================================================================

class TestMMEXOFASTFitResultsWithEmcee:
    """
    MMEXOFASTFitResults wrapping an EmceeLCFitter.

    format_results_as_df tests are moved here from the old
    EmceeFitResults (results.py) test class, since that class is removed.

    Key contract: the 'values' column for fitted parameters uses p50
    (median of post-burn-in chain), not best_theta (max-likelihood).
    best_theta is still accessible via fitter.results.x.
    """

    @pytest.fixture
    def fitter_data(self):
        return make_mock_emcee_fitter()

    @pytest.fixture
    def mock_emcee_fitter(self, fitter_data):
        return fitter_data[0]

    @pytest.fixture
    def parameters_to_fit(self, fitter_data):
        """Overrides module-level fixture: 7 default params, not 3."""
        return fitter_data[1]

    @pytest.fixture
    def best_dict(self, fitter_data):
        return fitter_data[2]

    @pytest.fixture
    def expected(self, fitter_data):
        return fitter_data[4]

    @pytest.fixture
    def full_result(self, mock_emcee_fitter):
        return MMEXOFASTFitResults(mock_emcee_fitter)

    @pytest.fixture
    def df(self, full_result):
        with patch(_MAG_FROM_FLUX_PATCH_PATH,
                   side_effect=_mock_get_mag_and_err_from_flux):
            return full_result.format_results_as_df()

    # --- get_params_from_results --------------------------------------------

    def test_get_params_excludes_chi2(self, full_result):
        assert 'chi2' not in full_result.get_params_from_results()

    def test_get_params_uses_linear_space_keys(self, full_result, parameters_to_fit):
        """Log-space keys (e.g. 'log_rho') must not appear; linear keys must."""
        params = full_result.get_params_from_results()
        for param in parameters_to_fit:
            linear_key = _get_parameter_name(param)
            assert linear_key in params, f"Linear key '{linear_key}' missing"
            if param != linear_key:
                assert param not in params, \
                    f"Log-space key '{param}' must not appear"

    def test_get_params_values_equal_best(self, full_result, best_dict):
        """get_params_from_results() values must match fitter.best (excl. chi2)."""
        params   = full_result.get_params_from_results()
        expected = {k: v for k, v in best_dict.items() if k != 'chi2'}
        assert set(params.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_almost_equal(params[key], expected[key])

    # --- get_sigmas_from_results --------------------------------------------

    def test_get_sigmas_keys_match_parameters_to_fit(
            self, full_result, parameters_to_fit):
        assert list(full_result.get_sigmas_from_results().keys()) == parameters_to_fit

    def test_get_sigmas_reads_from_results_sigmas(
            self, full_result, mock_emcee_fitter, parameters_to_fit):
        """get_sigmas_from_results() consumes fitter.results.sigmas."""
        sigmas   = full_result.get_sigmas_from_results()
        expected = dict(zip(parameters_to_fit, mock_emcee_fitter.results.sigmas))
        assert sigmas == pytest.approx(expected)

    def test_get_sigmas_values_are_mean_of_asymmetric(
            self, full_result, mock_emcee_fitter, parameters_to_fit):
        """Each sigma equals (sigma_minus + sigma_plus) / 2."""
        sigmas = full_result.get_sigmas_from_results()
        for i, param in enumerate(parameters_to_fit):
            expected = (
                mock_emcee_fitter.results.sigma_minus[i]
                + mock_emcee_fitter.results.sigma_plus[i]
            ) / 2
            assert sigmas[param] == pytest.approx(expected)

    def test_get_sigmas_are_positive(self, full_result):
        assert all(v > 0 for v in full_result.get_sigmas_from_results().values())

    # --- format_results_as_df: columns --------------------------------------

    def test_format_df_has_parameter_names_column(self, df):
        assert 'parameter_names' in df.columns

    def test_format_df_has_values_column(self, df):
        assert 'values' in df.columns

    def test_format_df_has_sigma_minus_column(self, df):
        assert 'sigma_minus' in df.columns

    def test_format_df_has_sigma_plus_column(self, df):
        assert 'sigma_plus' in df.columns

    def test_format_df_no_symmetric_sigmas_column(self, df):
        """Emcee DataFrame must not have a plain 'sigmas' column."""
        assert 'sigmas' not in df.columns

    # --- format_results_as_df: row count ------------------------------------

    def test_format_df_has_correct_row_count(self, df, parameters_to_fit):
        """n_fitted + chi2 + N_data + 2*n_datasets flux rows."""
        expected = (
            len(parameters_to_fit)   # 7 fitted
            + 1                      # chi2
            + 1                      # N_data
            + 2 * 2                  # source + blend for 2 datasets
        )
        assert len(df) == expected

    # --- format_results_as_df: fitted parameter rows ------------------------

    def test_format_fitted_params_all_present(self, df, parameters_to_fit):
        for param in parameters_to_fit:
            assert param in df['parameter_names'].values

    def test_format_fitted_param_values_are_p50(
            self, df, parameters_to_fit, expected):
        """Values for fitted rows must be p50 (not best_theta)."""
        for i, param in enumerate(parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(row['values'], expected['p50'][i])

    def test_format_fitted_param_sigma_minus_is_p50_minus_p16(
            self, df, parameters_to_fit, expected):
        for i, param in enumerate(parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_minus'], expected['sigma_minus'][i])

    def test_format_fitted_param_sigma_plus_is_p84_minus_p50(
            self, df, parameters_to_fit, expected):
        for i, param in enumerate(parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_plus'], expected['sigma_plus'][i])

    def test_format_fitted_sigma_minus_matches_results_object(
            self, df, mock_emcee_fitter, parameters_to_fit):
        """DataFrame sigma_minus matches fitter.results.sigma_minus (integration check)."""
        fitted_rows = df[df['parameter_names'].isin(parameters_to_fit)]
        np.testing.assert_array_almost_equal(
            fitted_rows['sigma_minus'].values,
            mock_emcee_fitter.results.sigma_minus,
        )

    def test_format_fitted_sigma_plus_matches_results_object(
            self, df, mock_emcee_fitter, parameters_to_fit):
        """DataFrame sigma_plus matches fitter.results.sigma_plus (integration check)."""
        fitted_rows = df[df['parameter_names'].isin(parameters_to_fit)]
        np.testing.assert_array_almost_equal(
            fitted_rows['sigma_plus'].values,
            mock_emcee_fitter.results.sigma_plus,
        )

    def test_format_fitted_sigma_minus_are_positive(self, df, parameters_to_fit):
        for param in parameters_to_fit:
            row = df[df['parameter_names'] == param].iloc[0]
            assert row['sigma_minus'] > 0

    def test_format_fitted_sigma_plus_are_positive(self, df, parameters_to_fit):
        for param in parameters_to_fit:
            row = df[df['parameter_names'] == param].iloc[0]
            assert row['sigma_plus'] > 0

    # --- format_results_as_df: chi2 row -------------------------------------

    def test_format_chi2_present(self, df):
        assert 'chi2' in df['parameter_names'].values

    def test_format_chi2_value(self, df, best_dict):
        row = df[df['parameter_names'] == 'chi2'].iloc[0]
        np.testing.assert_almost_equal(row['values'], best_dict['chi2'])

    def test_format_chi2_sigma_minus_is_nan(self, df):
        row = df[df['parameter_names'] == 'chi2'].iloc[0]
        assert np.isnan(row['sigma_minus'])

    def test_format_chi2_sigma_plus_is_nan(self, df):
        row = df[df['parameter_names'] == 'chi2'].iloc[0]
        assert np.isnan(row['sigma_plus'])

    # --- format_results_as_df: N_data row -----------------------------------

    def test_format_n_data_present(self, df):
        assert 'N_data' in df['parameter_names'].values

    def test_format_n_data_value(self, df):
        row = df[df['parameter_names'] == 'N_data'].iloc[0]
        assert row['values'] == _N_DATA

    def test_format_n_data_sigma_minus_is_nan(self, df):
        row = df[df['parameter_names'] == 'N_data'].iloc[0]
        assert np.isnan(row['sigma_minus'])

    def test_format_n_data_sigma_plus_is_nan(self, df):
        row = df[df['parameter_names'] == 'N_data'].iloc[0]
        assert np.isnan(row['sigma_plus'])

    # --- format_results_as_df: flux parameter rows --------------------------

    def test_format_flux_param_names_all_present(self, df):
        for name in _EXPECTED_FLUX_PARAM_NAMES:
            assert name in df['parameter_names'].values

    def test_format_flux_param_values_are_mock_magnitudes(self, df):
        for name, expected_mag in _EXPECTED_FLUX_PARAM_VALUES.items():
            row = df[df['parameter_names'] == name].iloc[0]
            np.testing.assert_almost_equal(row['values'], expected_mag)

    def test_format_flux_param_sigma_minus_is_nan(self, df):
        for name in _EXPECTED_FLUX_PARAM_NAMES:
            row = df[df['parameter_names'] == name].iloc[0]
            assert np.isnan(row['sigma_minus'])

    def test_format_flux_param_sigma_plus_is_nan(self, df):
        for name in _EXPECTED_FLUX_PARAM_NAMES:
            row = df[df['parameter_names'] == name].iloc[0]
            assert np.isnan(row['sigma_plus'])

    # --- format_results_as_df: row ordering ---------------------------------

    def test_format_fitted_params_before_fixed_params(self, df, parameters_to_fit):
        names = list(df['parameter_names'].values)
        for fitted in parameters_to_fit:
            for fixed in ['chi2', 'N_data']:
                assert names.index(fitted) < names.index(fixed), \
                    f"'{fitted}' must come before '{fixed}'"

    def test_format_fixed_params_before_flux_params(self, df):
        names = list(df['parameter_names'].values)
        for fixed in ['chi2', 'N_data']:
            for flux in _EXPECTED_FLUX_PARAM_NAMES:
                assert names.index(fixed) < names.index(flux), \
                    f"'{fixed}' must come before '{flux}'"


# ===========================================================================
# Group 4: MMEXOFASTFitResults wrapping EmceeLCFitter — fixed parameters
# ===========================================================================

class TestMMEXOFASTFitResultsWithEmceeFixedParams:
    """
    format_results_as_df when log_s and log_q are fixed (not sampled by emcee).

    Moved from old TestEmceeFitResultsWithFixedParams.
    """

    _PARAMETERS_TO_FIT = ['t_0', 'u_0', 't_E', 'log_rho', 'alpha']
    _FIXED_S = 1.1
    _FIXED_Q = 0.001

    @pytest.fixture
    def fitter_data(self):
        return make_mock_emcee_fitter(
            parameters_to_fit=self._PARAMETERS_TO_FIT,
            fixed_params_dict={'s': self._FIXED_S, 'q': self._FIXED_Q},
        )

    @pytest.fixture
    def mock_emcee_fitter(self, fitter_data):
        return fitter_data[0]

    @pytest.fixture
    def parameters_to_fit(self, fitter_data):
        return fitter_data[1]

    @pytest.fixture
    def expected(self, fitter_data):
        return fitter_data[4]

    @pytest.fixture
    def full_result(self, mock_emcee_fitter):
        return MMEXOFASTFitResults(mock_emcee_fitter)

    @pytest.fixture
    def df(self, full_result):
        with patch(_MAG_FROM_FLUX_PATCH_PATH,
                   side_effect=_mock_get_mag_and_err_from_flux):
            return full_result.format_results_as_df()

    # --- structure ----------------------------------------------------------

    def test_format_has_correct_row_count(self, df):
        """5 fitted + 3 fixed (s, q, chi2) + 1 N_data + 4 flux rows."""
        assert len(df) == (
            len(self._PARAMETERS_TO_FIT)   # 5
            + 3                            # s, q, chi2
            + 1                            # N_data
            + 2 * 2                        # 2 datasets
        )

    # --- fitted parameter rows ----------------------------------------------

    def test_format_fitted_params_all_present(self, df):
        for param in self._PARAMETERS_TO_FIT:
            assert param in df['parameter_names'].values

    def test_format_fitted_param_values_are_p50(self, df, expected):
        for i, param in enumerate(self._PARAMETERS_TO_FIT):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(row['values'], expected['p50'][i])

    def test_format_fitted_param_sigma_minus(self, df, expected):
        for i, param in enumerate(self._PARAMETERS_TO_FIT):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_minus'], expected['sigma_minus'][i])

    def test_format_fitted_param_sigma_plus(self, df, expected):
        for i, param in enumerate(self._PARAMETERS_TO_FIT):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_plus'], expected['sigma_plus'][i])

    # --- fixed parameter rows: s and q --------------------------------------

    def test_format_fixed_s_present(self, df):
        assert 's' in df['parameter_names'].values

    def test_format_fixed_q_present(self, df):
        assert 'q' in df['parameter_names'].values

    def test_format_fixed_s_value(self, df):
        row = df[df['parameter_names'] == 's'].iloc[0]
        np.testing.assert_almost_equal(row['values'], self._FIXED_S)

    def test_format_fixed_q_value(self, df):
        row = df[df['parameter_names'] == 'q'].iloc[0]
        np.testing.assert_almost_equal(row['values'], self._FIXED_Q)

    def test_format_fixed_s_sigma_minus_is_nan(self, df):
        assert np.isnan(df[df['parameter_names'] == 's'].iloc[0]['sigma_minus'])

    def test_format_fixed_s_sigma_plus_is_nan(self, df):
        assert np.isnan(df[df['parameter_names'] == 's'].iloc[0]['sigma_plus'])

    def test_format_fixed_q_sigma_minus_is_nan(self, df):
        assert np.isnan(df[df['parameter_names'] == 'q'].iloc[0]['sigma_minus'])

    def test_format_fixed_q_sigma_plus_is_nan(self, df):
        assert np.isnan(df[df['parameter_names'] == 'q'].iloc[0]['sigma_plus'])

    def test_format_log_s_absent(self, df):
        assert 'log_s' not in df['parameter_names'].values

    def test_format_log_q_absent(self, df):
        assert 'log_q' not in df['parameter_names'].values

    # --- row ordering -------------------------------------------------------

    def test_format_fitted_params_before_fixed_s_and_q(self, df):
        names = list(df['parameter_names'].values)
        for fitted in self._PARAMETERS_TO_FIT:
            for fixed in ['s', 'q']:
                assert names.index(fitted) < names.index(fixed), \
                    f"'{fitted}' must come before '{fixed}'"

    def test_format_fixed_s_and_q_before_flux_params(self, df):
        names = list(df['parameter_names'].values)
        for fixed in ['s', 'q']:
            for flux in _EXPECTED_FLUX_PARAM_NAMES:
                assert names.index(fixed) < names.index(flux), \
                    f"'{fixed}' must come before '{flux}'"


# ===========================================================================
# Group 5: FitRecord integration
# ===========================================================================

class TestFitRecordWithEmcee:
    """
    Integration tests for FitRecord built from MMEXOFASTFitResults(emcee_fitter).

    Old TestFitRecordWithEmcee used EmceeFitResults (results.py) directly;
    updated to use MMEXOFASTFitResults as the full_result wrapper.
    """

    @pytest.fixture
    def fit_key(self):
        return FitKey(
            lens_type=LensType.BINARY,
            source_type=SourceType.FINITE,
            parallax_branch=ParallaxBranch.NONE,
            lens_orb_motion=LensOrbMotion.NONE,
        )

    @pytest.fixture
    def fitter_data(self):
        return make_mock_emcee_fitter()

    @pytest.fixture
    def mock_emcee_fitter(self, fitter_data):
        return fitter_data[0]

    @pytest.fixture
    def parameters_to_fit(self, fitter_data):
        return fitter_data[1]

    @pytest.fixture
    def best_dict(self, fitter_data):
        return fitter_data[2]

    @pytest.fixture
    def expected(self, fitter_data):
        return fitter_data[4]

    @pytest.fixture
    def full_result(self, mock_emcee_fitter):
        return MMEXOFASTFitResults(mock_emcee_fitter)

    @pytest.fixture
    def record(self, full_result, fit_key):
        return FitRecord.from_full_result(
            model_key=fit_key,
            full_result=full_result,
        )

    @pytest.fixture
    def df(self, record):
        with patch(_MAG_FROM_FLUX_PATCH_PATH,
                   side_effect=_mock_get_mag_and_err_from_flux):
            return record.to_dataframe()

    # --- construction -------------------------------------------------------

    def test_from_full_result_returns_fit_record(self, record):
        assert isinstance(record, FitRecord)

    def test_record_is_complete(self, record):
        assert record.is_complete is True

    def test_record_params_not_none(self, record):
        assert record.params is not None

    def test_record_sigmas_not_none(self, record):
        assert record.sigmas is not None

    def test_record_params_equal_best(self, record, best_dict):
        """FitRecord.params must equal fitter.best (excluding chi2)."""
        expected = {k: v for k, v in best_dict.items() if k != 'chi2'}
        assert set(record.params.keys()) == set(expected.keys())
        for key in expected:
            np.testing.assert_almost_equal(record.params[key], expected[key])

    def test_record_full_result_is_mmexofast(self, record):
        assert isinstance(record.full_result, MMEXOFASTFitResults)

    def test_record_chi2_is_finite(self, record):
        assert np.isfinite(record.chi2())

    # --- to_dataframe -------------------------------------------------------

    def test_to_dataframe_has_sigma_minus_plus_columns(self, df):
        assert 'sigma_minus' in df.columns
        assert 'sigma_plus'  in df.columns

    def test_to_dataframe_has_no_symmetric_sigmas_column(self, df):
        assert 'sigmas' not in df.columns

    def test_to_dataframe_fitted_param_values_are_p50(
            self, df, parameters_to_fit, expected):
        """to_dataframe() values for fitted params must be p50."""
        for i, param in enumerate(parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(row['values'], expected['p50'][i])

    def test_to_dataframe_fitted_param_sigma_minus(
            self, df, parameters_to_fit, expected):
        for i, param in enumerate(parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_minus'], expected['sigma_minus'][i])

    def test_to_dataframe_fitted_param_sigma_plus(
            self, df, parameters_to_fit, expected):
        for i, param in enumerate(parameters_to_fit):
            row = df[df['parameter_names'] == param].iloc[0]
            np.testing.assert_almost_equal(
                row['sigma_plus'], expected['sigma_plus'][i])
