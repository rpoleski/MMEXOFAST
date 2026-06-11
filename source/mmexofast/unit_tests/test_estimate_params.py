import unittest
import numpy.testing
import os.path
import MulensModel
import numpy as np
import matplotlib.pyplot as plt

from mmexofast import estimate_params, com_trans, fitters
from mmexofast.config import DATA_PATH


class TestGetWideParams_alpha(unittest.TestCase):
    """
    Test that the model calculated from a given t_pl produces an
    anomaly at t_pl.

            - 't_0' (*float*): Time of maximum magnification.
            - 'u_0' (*float*): Impact parameter.
            - 't_E' (*float*): Einstein crossing time.
            - 't_pl' (*float*): Time at which to compute the wide model parameters.
            - 'dt' (*float*): Duration of the anomaly
            - 'dmag' (*float*): Magnitude difference of the perturbation

    """

    def setUp(self):
        self.params = {'t_0': 0., 'u_0': 0.5, 't_E': 15., 't_pl': 0, 'dt': 0.7, 'dmag': 0.5}

    def estimate_params(self, params):
        return estimate_params.get_wide_params(params)

    def do_test(self, t_pl):
        """
        make sure the magnification at t_pl is different for the estimated
        planet model than for a PSPL model.
        """
        params = {key: value for key, value in self.params.items()}
        params['t_pl'] = t_pl
        planet_params = self.estimate_params(params)
        if isinstance(planet_params, estimate_params.BinaryLensParams):
            planet_params = [planet_params]

        for params in planet_params:
            planet_model = MulensModel.Model(parameters=params.ulens)
            planet_model.set_magnification_methods([t_pl - 1., 'VBBL', t_pl + 1.])
            planet_model_mag = planet_model.get_magnification(t_pl)

            #print(planet_model)
            #plt.figure(figsize=(8, 4))
            #plt.suptitle(
            #    'u_0={0}, alpha={1:8.2f}'.format(params.ulens['u_0'], params.ulens['alpha']))
            #plt.subplot(1, 2, 1)
            ##plt.title('alpha = {0}'.format(planet_model.parameters.alpha))
            #planet_model.plot_trajectory(caustics=True)
            #
            #plt.subplot(1, 2, 2)
            #planet_model.plot_magnification()
            #plt.axvline(t_pl, color='black')
            #plt.show()

            pspl_model = MulensModel.Model(
                parameters={'t_0': self.params['t_0'], 'u_0': self.params['u_0'], 't_E': self.params['t_E']})
            pspl_model_mag = pspl_model.get_magnification(t_pl)

            #print(params.ulens)
            assert np.abs((planet_model_mag - pspl_model_mag) / pspl_model_mag) > 0.05

    def test_1(self):
        t_pl = self.params['t_0'] - 0.5 * self.params['t_E']
        self.do_test(t_pl)

    def test_2(self):
        t_pl = self.params['t_0'] - 1.5 * self.params['t_E']
        self.do_test(t_pl)

    def test_3(self):
        t_pl = self.params['t_0'] + 0.5 * self.params['t_E']
        self.do_test(t_pl)

    def test_4(self):
        t_pl = self.params['t_0'] + 1.5 * self.params['t_E']
        self.do_test(t_pl)


class TestGetCloseParams_alpha(TestGetWideParams_alpha):

    def estimate_params(self, params):
        return estimate_params.get_close_params(params)


class KB160625():
    """
    Parameters for KMT-2016-BLG-0625 Shin et al. 2024.
    """
    def __init__(self):
        # Section 3.4
        self.tau_pl = 0.609
        self.u_pl = 0.613
        self.t_pl = 7662.95
        self.t_E = 11.5
        self.s_close = 0.739
        self.s_wide = 1.352
        self.alpha = np.pi - np.array([0.12, 3.26])  # radians, alpha corrected to MMv3 system

        # Data from Table 5 (alpha corrected to MMv3 system):
        # s-
        self.close_params = {'t_0': 7655.951, 'u_0': 0.073, 't_E': 11.494,
                             's': 0.741, 'q': 2.357e-4, 'alpha': 180. - np.rad2deg(3.217),
                             'rho': 1.2256e-3}
        # s+
        self.wide_params = {'t_0': 7655.951, 'u_0': 0.075, 't_E':11.335,
                            's': 1.367, 'q': 0.727e-4, 'alpha': 180. - np.rad2deg(0.122), 'rho': 1.7656e-3}
        # 1L2S
        self.binary_source_params = {
            't_0_1': 7655.953, 'u_0_1': 0.078, 't_E': 10.946, 't_0_2': 7662.943, 'u_0_2': 3.751e-4, 'rho_2': 5.1309e-3, 'q_flux':  0.005
        }

        # dt and dmag estimated by-eye from figure in paper.
        self.dt = 0.25
        self.dmag = 0.3

        # Approximate values of t_0 and u_0
        self.t_0 = 7655.95
        self.u_0 = 0.075

        self.tol = 0.03  # 2% uncertainty based on variation in u0

        self.params = {
            't_0': self.t_0, 'u_0': self.u_0, 't_E': self.t_E, 't_pl': self.t_pl,
            'dt': self.dt, 'dmag': self.dmag
            }


class OB180383():
    """
    Parameters for OGLE-2018-BLG-0383 Wang et al. 2022.
    """
    def __init__(self):
        self.dmag = -0.07  # preamble to Section 3.2

        # Section 3.2.1 Heuristic analysis
        self.pspl_est = {'t_0': 8199.2, 'u_0': 0.071, 't_E': 11.3}  # Eq. 8
        self.t_0, self.u_0, self.t_E = self.pspl_est['t_0'], self.pspl_est['u_0'], self.pspl_est['t_E']
        self.tau_pl, self.u_pl, self.alpha_est = -2.04, 2.05, 1.98  # deg, Eq. 9, u_anom --> u_pl
        self.alpha = np.deg2rad(self.alpha_est)  # might need to check reflections/conversion to MMv3 system
        self.t_pl = self.pspl_est['t_0'] + self.tau_pl * self.pspl_est['t_E']  # derived
        self.s_wide, self.s_close = 2.46, 0.41  # Eq. 10: s_plus, s_minus
        self.dt = 2. * 0.55  # 2. * t_fwhm
        self.rho_est = 0.024  # Eq. 12
        self.delta_A = 0.61  # Eq. 13
        self.q_est = 1.8e-4  # Eq. 14

        self.params = {
            't_0': self.t_0, 'u_0': self.u_0, 't_E': self.t_E, 't_pl': self.t_pl,
            'dt': self.dt, 'dmag': self.dmag
            }
        self.tol = 0.03  # 2% uncertainty based on variation in u0 for 1S solutions. (incr to 7% for 2S soln)

        # Table 2: Fitted parameters
        pspl = {'t_0': 8199.244, 'u_0': 0.072, 't_E': 11.15}
        self.wide_params = {'t_0': 8199.239, 'u_0': 0.071, 't_E': 11.35, 'rho': 0.0238, 'alpha': 181.98, 's': 2.453, 'q': 2.14e-4}
        close_upper = {
            't_0': 8199.247, 'u_0': 0.071, 't_E': 11.34, 'rho': 0.0060, 'alpha': 355.86, 's': 0.405, 'q': 23.6e-4}
        close_lower = {
            't_0': 8199.247, 'u_0': 0.072, 't_E': 11.46, 'rho': 0.0056, 'alpha': 7.84, 's': 0.404, 'q': 21.5e-4}
        binary_source = {
            't_0_1': 8199.244, 't_0_2': 8176.022, 'u_0_1': 0.074, 'u_0_2': 0.0007, 't_E': 11.34,
            'rho_1': 0.058, 'rho_2': 0.0202, 'q_flux_I': 0.0057}


class TestParameterEstimatorKB160625(unittest.TestCase, KB160625):

    def setUp(self):
        KB160625.__init__(self)
        self.estimator = estimate_params.ParameterEstimator(self.params, limit='point')
        #self.ulens_params = estimate_params.get_wide_params(self.params, limit='point')

    def test_correct_alpha(self):
        expected = [20., 20., -20., -20.]
        input = [20., 380., -20, -740.]
        for value_exp, value_in in zip(expected, input):
            print('expected', value_exp, 'input', value_in)
            numpy.testing.assert_almost_equal(value_exp, self.estimator._correct_alpha(value_in))

    def test_get_rho_dwarf(self):
        estimator = estimate_params.ParameterEstimator(self.params, limit='dwarf')
        assert estimator.get_rho() == 0.001

    def test_get_rho_giant(self):
        estimator = estimate_params.ParameterEstimator(self.params, limit='giant')
        assert estimator.get_rho() == 0.05

    def test_get_rho_point(self):
        estimator = estimate_params.ParameterEstimator(self.params, limit='point')
        assert estimator.get_rho() is None

    def test_get_rho_error(self):
        with self.assertRaises(ValueError):
            estimator = estimate_params.ParameterEstimator(self.params)
            estimator.get_rho()

    def test_t_0(self):
        assert self.estimator.t_0 == self.params['t_0']

    def test_u_0(self):
        assert self.estimator.u_0 == self.params['u_0']

    def test_t_E(self):
        assert self.estimator.t_E == self.params['t_E']

    def test_tau_pl(self):
        np.testing.assert_allclose(self.estimator.tau_pl, self.tau_pl, rtol=self.tol)

    def test_u_pl(self):
        np.testing.assert_allclose(self.estimator.u_pl, self.u_pl, rtol=self.tol)

    def test_alpha(self):
        if isinstance(self.alpha, float):
            np.testing.assert_allclose(self.estimator.alpha, np.rad2deg(self.alpha), rtol=self.tol)
        else:
            index = np.argmin(np.abs(self.estimator.alpha - np.rad2deg(self.alpha)))
            np.testing.assert_allclose(self.estimator.alpha, np.rad2deg(self.alpha[index]), rtol=self.tol)

    def test_rho(self):
        assert self.estimator.rho is None

    def test_rho_manual(self):
        estimator = estimate_params.ParameterEstimator(self.params)
        estimator.rho = 0.3
        assert estimator.rho == 0.3


class TestParameterEstimatorOB180383(TestParameterEstimatorKB160625, OB180383):

    def setUp(self):
        OB180383.__init__(self)
        self.estimator = estimate_params.ParameterEstimator(self.params, limit='point')


class TestWideParameterEstimatorOB180383(TestParameterEstimatorOB180383, OB180383):

    def setUp(self):
        OB180383.__init__(self)
        self.estimator = estimate_params.WidePlanetParameterEstimator(self.params, limit='GG97')

    def test_rho(self):
        np.testing.assert_allclose(self.estimator.rho, self.rho_est, rtol=self.tol)

    def test_s(self):
        np.testing.assert_allclose(self.estimator.s, self.s_wide, rtol=self.tol)

    def test_q_manual_dA(self):
        estimator = estimate_params.WidePlanetParameterEstimator(self.params, limit='GG97')
        estimator._delta_A = self.delta_A
        np.testing.assert_allclose(estimator.q, self.q_est, rtol=self.tol)

    def test_q(self):
        self.skipTest('this test fails because of the large negative blending in the event, which affects the calculation of delta_A.')
        np.testing.assert_allclose(self.estimator.q, self.q_est, rtol=self.tol)

    def test_delta_A(self):
        self.skipTest('this test fails because of the large negative blending in the event, which affects the calculation of delta_A.')
        np.testing.assert_allclose(self.estimator.delta_A, self.delta_A, rtol=self.tol)


class TestCloseUpperParameterEstimatorOB180383(TestParameterEstimatorOB180383, OB180383):

    def setUp(self):
        OB180383.__init__(self)
        self.estimator = estimate_params.CloseUpperBinaryParameterEstimator(self.params, limit='point')

    def test_s(self):
        np.testing.assert_allclose(self.estimator.s, self.s_close, rtol=self.tol)

    def test_q(self):
        np.testing.assert_allclose(self.estimator.q, 0.004, rtol=self.tol)

    def test_alpha(self):
        pass


class TestCloseLowerParameterEstimatorOB180383(TestCloseUpperParameterEstimatorOB180383, OB180383):

    def setUp(self):
        OB180383.__init__(self)
        self.estimator = estimate_params.CloseLowerBinaryParameterEstimator(self.params, limit='point')


class TestGetWideParams(unittest.TestCase, KB160625):

    def setUp(self):
        KB160625.__init__(self)
        self.ulens_params = estimate_params.get_wide_params(self.params)

    def test_pspl(self):
        """
        t_0, u_0, t_E
        """
        for key in ['t_0', 'u_0', 't_E']:
            assert self.ulens_params.ulens[key] == self.__getattribute__(key)

    def test_s(self):
        np.testing.assert_allclose(self.ulens_params.ulens['s'], self.s_wide, rtol=self.tol)

    def test_alpha(self):
        # self.alpha has 2 values. choose the one closest to the results.
        if isinstance(self.alpha, float):
            np.testing.assert_allclose(self.ulens_params.ulens['alpha'], np.rad2deg(self.alpha), rtol=self.tol)
        else:
            index = np.argmin(np.abs(self.ulens_params.ulens['alpha'] - np.rad2deg(self.alpha)))
            np.testing.assert_allclose(self.ulens_params.ulens['alpha'], np.rad2deg(self.alpha[index]), rtol=self.tol)

    def test_mag_methods(self):
        expected_values = [self.t_pl - 5. * self.dt / 2., 'VBBL', self.t_pl + 5. * self.dt / 2.]
        for actual, expected in zip(self.ulens_params.mag_methods[4:7], expected_values):
            if isinstance(actual, str):
                assert actual == expected
            else:
                np.testing.assert_allclose(actual, expected, atol=0.001)


class TestGetWideParams2(TestGetWideParams, OB180383):

    def setUp(self):
        OB180383.__init__(self)
        self.ulens_params = estimate_params.get_wide_params(self.params)

    def test_rho(self):
        # Gould & Gaucherel approximation
        # Ap = 2(q / ρ^2)
        np.testing.assert_allclose(self.ulens_params.ulens['rho'], self.rho_est, rtol=self.tol)

        # These tests don't work because of negative blending.
        #Ap_est = 2. * self.ulens_params.ulens['q'] / self.ulens_params.ulens['rho']**2
        #np.testing.assert_allclose(Ap_est, self.delta_A, rtol=self.tol)
        #
        #np.testing.assert_allclose(self.ulens_params.ulens['q'], self.q_est, rtol=self.tol)


class TestGetCloseParams(unittest.TestCase, KB160625):

    def setUp(self):
        KB160625.__init__(self)
        self.ulens_params = estimate_params.get_close_params(self.params, q=self)
        

class TestAnomalyParameterEstimator(unittest.TestCase):

    def setUp(self):
        datafile = os.path.join(DATA_PATH, 'unit_test_data', 'planet4AF.dat')
        self.data = MulensModel.MulensData(
            file_name=datafile,
            phot_fmt='mag')
        self.true_params, self.input_fluxes = self._parse_header(datafile)
        new_params = com_trans.co_mass_to_co_magnif(self.true_params)
        new_params['t_E'] = self.true_params['t_E']
        self.pspl_params = new_params
        self.af_results = {'t_0': 17.43489583333333, 't_eff': 0.421875, 'j': 2.0, 'chi2': 98.97724735834696,
                           'dchi2_zero': 218.83573427369782, 'dchi2_flat': 143.937564049782}

    def _parse_header(self, datafile):
        with open(datafile, 'r') as file_:
            lines = file_.readlines()

        elements = lines[0].split()
        ulens_params = {}
        for i, element in enumerate(elements):
            if element[-2:] == "':":
                key = element.strip('{')[1:-2]
                ulens_params[key] = float(elements[i + 1].strip(',').strip('}'))

        elements = lines[1].split()
        fluxes = {}
        for i, element in enumerate(elements):
            if element == '=':
                fluxes[elements[i - 1]] = float(elements[i + 1].strip(','))

        return ulens_params, fluxes

    def test_update_pspl_model(self):
        self.skipTest('JCY: this test does not work. Maybe there is a change in origin. Maybe it is a bad test, regardless.')
        fitter = fitters.SFitFitter(datasets=[self.data], initial_model=self.pspl_params)
        fitter.run()
        test_pspl = {key: fitter.best[key] for key in self.pspl_params.keys()}
        estimator = estimate_params.AnomalyPropertyEstimator(
            datasets=self.data, pspl_params=test_pspl, af_results=self.af_results
        )
        estimator.update_pspl_model()

        print(test_pspl)
        print(self.pspl_params)
        print(estimator.refined_pspl_params)
        #event = MulensModel.Event(datasets=self.data, model=MulensModel.Model(parameters=test_pspl))
        #event.plot()
        #
        #event_2 = MulensModel.Event(datasets=estimator.masked_datasets, model=MulensModel.Model(parameters=estimator.refined_pspl_params))
        #event_2.plot(show_bad=True)
        #plt.show()

        for key, value in estimator.refined_pspl_params.items():
            np.testing.assert_allclose(value, self.pspl_params[key], rtol=0.001)

    def test_get_anomaly_lc_params(self):
        estimator = estimate_params.AnomalyPropertyEstimator(
            datasets=self.data, pspl_params=self.pspl_params, af_results=self.af_results
        )
        results = estimator.get_anomaly_lc_parameters()
        expected = {'t_pl': 17.44, 'dt': 0.3, 'dmag': -0.12}
        print(results)
        for key, value in expected.items():
            if key == 'dt':
                np.testing.assert_allclose(results[key], value, rtol=2)
            else:
                np.testing.assert_allclose(results[key], value, rtol=0.2)


def test_model_pspl_at_pl():
    raise unittest.SkipTest()


class TestBinarySourceParams(unittest.TestCase):

    def test_set_source_flux_ratio(self):
        self.skipTest('Not Implemented')


def test_get_binary_source_params():
    raise unittest.SkipTest()


class TestBinaryLensParams(unittest.TestCase):
    """
    Unit tests for BinaryLensParams using known parameters from
    published microlensing events.

    Test cases
    ----------
    KB160625 : KMT-2016-BLG-0625 (Shin et al. 2024), wide topology.
    OB180383 : OGLE-2018-BLG-0383 (Wang et al. 2022), wide topology.

    Notes
    -----
    Refined BinaryLensParams instances are built once in setUpClass to
    avoid repeated VBBL calls across tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Build initial and refined BinaryLensParams instances for each
        test event. VBBL evaluations are expensive; compute once for
        the whole class.
        """
        kb = KB160625()
        ob = OB180383()

        cls.test_cases = {
            'KB160625': {
                'ulens': kb.wide_params,
                'params': kb.params,
            },
            'OB180383': {
                'ulens': ob.wide_params,
                'params': ob.params,
            },
        }

        cls.blp_refined = {}
        for name, case in cls.test_cases.items():
            blp = estimate_params.BinaryLensParams(case['ulens'])
            blp.set_mag_method(case['params'])
            blp.refine_mag_methods()
            cls.blp_refined[name] = blp

    def _make_fresh_blp(self, event_name):
        """
        Return a BinaryLensParams with set_mag_method called but not
        yet refined, for the named event.
        """
        case = self.test_cases[event_name]
        blp = estimate_params.BinaryLensParams(case['ulens'])
        blp.set_mag_method(case['params'])
        return blp

    # ------------------------------------------------------------------ #
    # set_mag_method
    # ------------------------------------------------------------------ #

    def test_set_mag_method_structure(self):
        """mag_methods should have 11 elements with method strings at
        odd indices."""
        expected_strings = [
            'point_source', 'hexadecapole', 'VBBL', 'hexadecapole', 'point_source'
        ]
        for name in self.test_cases:
            with self.subTest(event=name):
                blp = self._make_fresh_blp(name)
                self.assertEqual(len(blp.mag_methods), 11)
                self.assertEqual(blp.mag_methods[1::2], expected_strings)

    def test_set_mag_method_stores_params(self):
        """set_mag_method should store params as self.params."""
        for name, case in self.test_cases.items():
            with self.subTest(event=name):
                blp = self._make_fresh_blp(name)
                self.assertIs(blp.params, case['params'])

    def test_set_mag_method_boundary_formulas(self):
        """Initial boundary times should match the defining formulas exactly."""
        for name, case in self.test_cases.items():
            with self.subTest(event=name):
                blp = self._make_fresh_blp(name)
                p = case['params']
                t_E = p['t_E']
                t_0 = p['t_0']
                t_pl = p['t_pl']
                t_star = p['dt'] / 2.
                expected = [
                    np.min((t_0 - t_E, t_pl - t_E / 2., t_pl - 20. * t_star)),
                    t_pl - 10. * t_star,
                    t_pl - 5. * t_star,
                    t_pl + 5. * t_star,
                    t_pl + 10. * t_star,
                    np.max((t_0 + t_E, t_pl + t_E / 2., t_pl + 20. * t_star)),
                ]
                for i, (exp, act) in enumerate(zip(expected, blp.mag_methods[0::2])):
                    self.assertAlmostEqual(
                        act, exp, places=10,
                        msg=f"Time boundary at index {2*i} mismatch: "
                            f"expected {exp:.6f}, got {act:.6f}")

    def test_set_mag_method_boundaries_monotonic(self):
        """Initial boundary times should be strictly increasing."""
        for name in self.test_cases:
            with self.subTest(event=name):
                blp = self._make_fresh_blp(name)
                times = blp.mag_methods[0::2]
                for i in range(len(times) - 1):
                    self.assertLess(
                        times[i], times[i + 1],
                        msg=f"Times not monotonic at positions {i} and {i+1}: "
                            f"{times[i]:.4f} >= {times[i+1]:.4f}")

    # ------------------------------------------------------------------ #
    # refine_mag_methods: structural
    # ------------------------------------------------------------------ #

    def test_refine_raises_before_set_mag_method(self):
        """refine_mag_methods should raise RuntimeError if set_mag_method
        has not been called."""
        case = next(iter(self.test_cases.values()))
        blp = estimate_params.BinaryLensParams(case['ulens'])
        with self.assertRaises(RuntimeError):
            blp.refine_mag_methods()

    def test_hard_limits_unchanged_after_refinement(self):
        """The outer hard limits (indices 0 and 10) should not be
        modified by refine_mag_methods."""
        for name in self.test_cases:
            with self.subTest(event=name):
                blp_init = self._make_fresh_blp(name)
                blp_ref = self.blp_refined[name]
                self.assertEqual(
                    blp_ref.mag_methods[0], blp_init.mag_methods[0],
                    msg="t_start (index 0) changed during refinement")
                self.assertEqual(
                    blp_ref.mag_methods[10], blp_init.mag_methods[10],
                    msg="t_end (index 10) changed during refinement")

    def test_refined_boundaries_monotonic(self):
        """Refined boundary times should remain strictly increasing."""
        for name in self.test_cases:
            with self.subTest(event=name):
                times = self.blp_refined[name].mag_methods[0::2]
                for i in range(len(times) - 1):
                    self.assertLess(
                        times[i], times[i + 1],
                        msg=f"Refined times not monotonic at positions "
                            f"{i} and {i+1}: "
                            f"{times[i]:.4f} >= {times[i+1]:.4f}")

    def test_refined_boundaries_within_hard_limits(self):
        """All refined inner boundaries should lie within the outer
        hard limits."""
        for name in self.test_cases:
            with self.subTest(event=name):
                blp = self.blp_refined[name]
                t_start = blp.mag_methods[estimate_params.BinaryLensParams._T_START_IDX]
                t_end = blp.mag_methods[estimate_params.BinaryLensParams._T_END_IDX]
                for t in blp.mag_methods[2:-1:2]:  # indices 2, 4, 6, 8
                    self.assertGreaterEqual(
                        t, t_start,
                        msg=f"Refined boundary t={t:.4f} < hard limit "
                            f"t_start={t_start:.4f}")
                    self.assertLessEqual(
                        t, t_end,
                        msg=f"Refined boundary t={t:.4f} > hard limit "
                            f"t_end={t_end:.4f}")

    def test_method_strings_unchanged_after_refinement(self):
        """Method name strings should not be altered by
        refine_mag_methods."""
        expected = [
            'point_source', 'hexadecapole', 'VBBL', 'hexadecapole', 'point_source'
        ]
        for name in self.test_cases:
            with self.subTest(event=name):
                self.assertEqual(
                    self.blp_refined[name].mag_methods[1::2], expected)

    # ------------------------------------------------------------------ #
    # Boundary consistency: primary physical test
    # ------------------------------------------------------------------ #
    def test_boundary_consistency(self):
        """
        At each refined boundary t_b, the less precise method evaluated
        just outside and the more precise method evaluated just inside
        should agree within a magnification-dependent threshold.

        With direction = sign(t_b - t_pl):
          t_outside = t_b + direction * epsilon  (approx method territory)
          t_inside  = t_b - direction * epsilon  (precise method territory)

        epsilon = 1e-7 days is far smaller than the brentq xtol of 0.01
        days, so this effectively tests agreement at the boundary itself.

        Threshold rule (base = 0.001):
          A <  3 : relative, threshold = 0.001 * mag_precise  (0.1%)
          A >= 3 : absolute, threshold = 0.001
        """
        epsilon = 1e-7

        boundary_specs = [
            (estimate_params.BinaryLensParams._T_HEXA_LEFT_IDX,
             'hexadecapole', 'point_source', 'hexa/ps left'),
            (estimate_params.BinaryLensParams._T_VBBL_LEFT_IDX,
             'VBBL', 'hexadecapole', 'VBBL/hexa left'),
            (estimate_params.BinaryLensParams._T_VBBL_RIGHT_IDX,
             'VBBL', 'hexadecapole', 'VBBL/hexa right'),
            (estimate_params.BinaryLensParams._T_HEXA_RIGHT_IDX,
             'hexadecapole', 'point_source', 'hexa/ps right'),
        ]

        for event_name, case in self.test_cases.items():
            blp = self.blp_refined[event_name]
            t_pl = float(case['params']['t_pl'])
            models = {
                'VBBL': blp._make_model('VBBL'),
                'hexadecapole': blp._make_model('hexadecapole'),
                'point_source': blp._make_model('point_source'),
            }

            for idx, precise_name, approx_name, boundary_name in boundary_specs:
                with self.subTest(event=event_name, boundary=boundary_name):
                    t_b = float(blp.mag_methods[idx])
                    direction = float(np.sign(t_b - t_pl))
                    t_outside = t_b + direction * epsilon
                    t_inside = t_b - direction * epsilon

                    mag_outside = float(models[approx_name].get_magnification(t_outside))
                    mag_inside = float(models[precise_name].get_magnification(t_inside))
                    diff = abs(mag_outside - mag_inside)
                    threshold = estimate_params.BinaryLensParams._mag_threshold(mag_inside)

                    self.assertLessEqual(
                        diff, threshold,
                        msg=f"Discontinuity at {boundary_name} "
                            f"(t={t_b:.4f}, A={mag_inside:.3f}): "
                            f"|{approx_name}(outside) - "
                            f"{precise_name}(inside)| = "
                            f"{diff:.6f} > threshold {threshold:.6f}")

    # ------------------------------------------------------------------ #
    # _make_model
    # ------------------------------------------------------------------ #

    def test_make_model_default_method(self):
        """_make_model should return a Model with the correct default
        magnification method."""
        case = next(iter(self.test_cases.values()))
        blp = estimate_params.BinaryLensParams(case['ulens'])
        for method in ('VBBL', 'hexadecapole', 'point_source'):
            with self.subTest(method=method):
                model = blp._make_model(method)
                self.assertEqual(model.default_magnification_method, method)


if __name__ == '__main__':
    unittest.main()

