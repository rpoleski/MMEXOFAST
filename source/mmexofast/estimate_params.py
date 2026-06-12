#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
Classes and functions for estimating initial parameters for microlensing models.

Provides tools for estimating starting parameters for PSPL from the results of
EventFinderGridSearch and for wide binary lens, close binary lens, and binary
source models given the properties of a detected photometric anomaly. Includes
grid search with optional Nelder-Mead refinement for binary lens parameter
estimation, and ensemble initialization for MCMC sampling.
"""

# Created by Luca Campiani in January 2024
# Updated by Jennifer Yee, May 2025
from itertools import product
import pandas as pd
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.signal import find_peaks
import scipy.stats
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import warnings
# import copy

import MulensModel
from .mulens_object_config import ModelConfig, EventConfig


def get_PSPL_params(ef_grid_point, datasets, model_config=None, event_config=None, verbose=False):
    """
    Estimate initial PSPL parameters by grid search over u_0 and t_E.

    Parameters
    ----------
    ef_grid_point : dict
        Best point from the EventFinder grid search. Must contain ``'t_0'``.
    datasets : list of MulensModel.MulensData
        Photometric datasets to evaluate chi2 against.
    model_config : ModelConfig, optional
        Configuration for Model construction. If None, a default
        ``ModelConfig`` is used (no coords, no limb darkening).
    event_config : EventConfig, optional
        Configuration for Event construction. If None, a default
        ``EventConfig`` is used (no coords, no flux fixing).
    verbose : bool, optional
        If True, print progress. Default is False.

    Returns
    -------
    dict
        Best-fit parameter dictionary with keys ``'t_0'``, ``'u_0'``,
        ``'t_E'``.
    """
    _model_config = model_config if model_config is not None else ModelConfig()
    _event_config = event_config if event_config is not None else EventConfig()

    t_0s = ef_grid_point['t_0'] + ef_grid_point['t_eff'] * np.linspace(-1, 1, 7)
    u_0s = [0.01, 0.1, 0.3, 1.0, 1.5]
    t_Es = [1., 3., 10., 20., 40., 100.]
    best_chi2 = np.inf
    best_params = None
    for t_0, u_0, t_E in product(t_0s, u_0s, t_Es):
        params = {'t_0': t_0, 't_E': t_E, 'u_0': u_0}
        model = _model_config.build(parameters=params)
        event = _event_config.build(model=model, datasets=datasets)
        if event.get_chi2() < best_chi2:
            best_params = params
            best_chi2 = event.chi2

    return best_params


class BinaryLensParams():
    """
    Container for binary lens model parameters and magnification methods.

    Attributes
    ----------
    ulens : dict
        Binary lens parameter dictionary suitable for passing to
        ``MulensModel.Model``.
    mag_methods : list or None
        Magnification methods in MulensModel convention. Set by
        :meth:`set_mag_method` and optionally refined by
        :meth:`refine_mag_methods`.
    params : dict or None
        Anomaly light curve parameters. Stored by :meth:`set_mag_method`
        for use by :meth:`refine_mag_methods`.
    """

    # Index map for mag_methods list:
    # [t_start, 'point_source', t1, 'hexadecapole', t2, 'VBBL',
    #  t3, 'hexadecapole', t4, 'point_source', t_end]
    #    0           1       2        3          4     5
    #    6       7           8       9           10
    _T_START_IDX = 0
    _T_HEXA_LEFT_IDX = 2
    _T_VBBL_LEFT_IDX = 4
    _T_VBBL_RIGHT_IDX = 6
    _T_HEXA_RIGHT_IDX = 8
    _T_END_IDX = 10

    def __init__(self, ulens):
        self.ulens = ulens
        self.mag_methods = None
        self.params = None

    def set_mag_method(self, params):
        """
        Set the magnification calculation method based on input parameters.

        Sets up a sequence of magnification methods transitioning from
        point_source to hexadecapole to VBBL and back, centered on the
        anomaly time. Also stores ``params`` as ``self.params`` for use
        by :meth:`refine_mag_methods`.

        Parameters
        ----------
        params : dict
            Anomaly light curve parameters as returned by
            :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
            Required keys:

            - ``'t_0'`` : float, time of maximum magnification.
            - ``'u_0'`` : float, impact parameter.
            - ``'t_E'`` : float, Einstein crossing time.
            - ``'t_pl'`` : float, time of the anomaly.
            - ``'dt'`` : float, duration of the anomaly.
            - ``'dmag'`` : float, magnitude difference of the perturbation.

        Returns
        -------
        None
        """
        self.params = params
        t_E = params['t_E']
        t_0 = params['t_0']
        t_pl = params['t_pl']
        t_star = params['dt'] / 2.
        self.mag_methods = [
            np.min((t_0 - t_E, t_pl - t_E / 2., t_pl - 20. * t_star)),
            'point_source',
            t_pl - 10. * t_star,
            'hexadecapole',
            t_pl - 5. * t_star,
            'VBBL',
            t_pl + 5. * t_star,
            'hexadecapole',
            t_pl + 10. * t_star,
            'point_source',
            np.max((t_0 + t_E, t_pl + t_E / 2., t_pl + 20. * t_star))]

    @staticmethod
    def _mag_threshold(mag_precise, base=0.0001):
        """
        Return the acceptable absolute difference in magnification between
        two methods.

        For low magnification (A < 3), applies a relative threshold to
        avoid over-constraining the boundary in regions where the absolute
        difference between methods is naturally small. For high
        magnification (A >= 3), applies a fixed absolute threshold since
        relative differences can become unphysically tight near caustics.

        Parameters
        ----------
        mag_precise : float
            Magnification from the more precise model at the evaluation
            point. Used to determine which regime applies and to scale the
            relative threshold.
        base : float, optional
            Base precision level. Controls both the relative precision
            (as a fraction) for A < 3 and the absolute threshold for
            A >= 3. Default 0.01 (i.e. 1%).

        Returns
        -------
        float
            Absolute threshold on ``|mag_precise - mag_approx|``.
        """
        if mag_precise >= 3.0:
            return base                  # absolute
        else:
            return base * mag_precise    # relative

    def _make_model(self, default_method):
        """
        Build a ``MulensModel.Model`` from ``self.ulens`` with a fixed
        default magnification method.

        Parameters
        ----------
        default_method : str
            One of ``'VBBL'``, ``'hexadecapole'``, or ``'point_source'``.

        Returns
        -------
        MulensModel.Model
        """
        model = MulensModel.Model(self.ulens)
        model.default_magnification_method = default_method
        return model

    def _find_method_boundary(self, idx, model_precise, model_approx,
                               base=0.0001, xtol=0.01):
        """
        Find the refined boundary time at ``self.mag_methods[idx]``.

        Checks whether ``model_precise`` and ``model_approx`` agree to
        within the magnification-dependent threshold (see
        :meth:`_mag_threshold`) at the current boundary. If they already
        agree, the boundary is returned unchanged. Otherwise, uses
        exponential search to find an outer bracket where the methods
        agree, then ``scipy.optimize.brentq`` to locate the transition
        precisely.

        The search direction is determined by the sign of
        ``mag_methods[idx] - t_pl``: left boundaries search further left,
        right boundaries search further right. The hard limits are
        ``mag_methods[_T_START_IDX]`` and ``mag_methods[_T_END_IDX]``.

        Parameters
        ----------
        idx : int
            Index into ``self.mag_methods`` of the boundary time to
            refine. Must be one of the numeric (time) entries, i.e. one
            of ``_T_HEXA_LEFT_IDX``, ``_T_VBBL_LEFT_IDX``,
            ``_T_VBBL_RIGHT_IDX``, or ``_T_HEXA_RIGHT_IDX``.
        model_precise : MulensModel.Model
            The more precise magnification method model (e.g. VBBL).
        model_approx : MulensModel.Model
            The less precise magnification method model (e.g.
            hexadecapole).
        base : float, optional
            Base precision level passed to :meth:`_mag_threshold`.
            Default 0.01.
        xtol : float, optional
            Time precision in days passed to ``brentq``. Default 0.01.

        Returns
        -------
        float
            Refined boundary time. Equal to the original
            ``mag_methods[idx]`` if the methods already agree there, or
            the appropriate hard limit if the exponential search reaches
            it without finding agreement.

        Warns
        -----
        UserWarning
            If the exponential search reaches the hard limit before
            finding a bracket where the methods agree.
        """
        t_start = float(self.mag_methods[idx])  # ensure scalars
        t_pl = float(self.params['t_pl'])
        direction = np.sign(t_start - t_pl)

        t_limit = (self.mag_methods[self._T_START_IDX] if direction < 0
                   else self.mag_methods[self._T_END_IDX])

        _cache = {}

        def mag_diff(t):
            if t not in _cache:
                mag_p = float(model_precise.get_magnification(t))
                mag_a = float(model_approx.get_magnification(t))
                threshold = self._mag_threshold(mag_p, base=base)
                _cache[t] = abs(mag_p - mag_a) - threshold
            return _cache[t]

        # Guard: methods already agree at the initial boundary
        if mag_diff(t_start) <= 0:
            return float(t_start)

        # Phase 1: exponential search for a bracket where methods agree
        step = abs(t_start - t_pl)
        t_outer = t_start
        while True:
            t_outer += direction * step
            if direction * t_outer >= direction * t_limit:
                warnings.warn(
                    f"Reached hard limit t={t_limit:.3f} at "
                    f"mag_methods[{idx}]; methods may still disagree."
                )
                return float(t_limit)
            if mag_diff(t_outer) <= 0:
                break
            step *= 2.0

        # Phase 2: brentq to locate the transition precisely
        return float(brentq(
            mag_diff,
            min(t_start, t_outer),
            max(t_start, t_outer),
            xtol=xtol))

    def _boundaries_monotonic(self):
        """
        Check whether all time values in ``self.mag_methods`` are strictly
        increasing.

        Returns
        -------
        bool
            True if ``mag_methods[0] < mag_methods[2] < ... < mag_methods[10]``.
        """
        times = self.mag_methods[0::2]
        return all(times[i] < times[i + 1] for i in range(len(times) - 1))

    def _apply_refinement(self, base, xtol, model_vbbl, model_hexa, model_ps):
        """
        Run one pass of boundary refinement for all four method transitions.

        Updates ``self.mag_methods`` in place at the four inner boundary
        indices. Called by :meth:`refine_mag_methods` with different
        ``base`` values as part of its fallback sequence.

        Parameters
        ----------
        base : float
            Base precision level passed to :meth:`_find_method_boundary`.
        xtol : float
            Time precision in days passed to :meth:`_find_method_boundary`.
        model_vbbl : MulensModel.Model
            Model with ``default_magnification_method = 'VBBL'``.
        model_hexa : MulensModel.Model
            Model with ``default_magnification_method = 'hexadecapole'``.
        model_ps : MulensModel.Model
            Model with ``default_magnification_method = 'point_source'``.

        Returns
        -------
        None
        """
        boundaries = [
            (self._T_HEXA_LEFT_IDX,  model_hexa, model_ps),
            (self._T_VBBL_LEFT_IDX,  model_vbbl, model_hexa),
            (self._T_VBBL_RIGHT_IDX, model_vbbl, model_hexa),
            (self._T_HEXA_RIGHT_IDX, model_hexa, model_ps),
        ]
        for idx, model_precise, model_approx in boundaries:
            self.mag_methods[idx] = self._find_method_boundary(
                idx, model_precise, model_approx, base, xtol)

    def refine_mag_methods(self, base=0.0001, xtol=0.01):
        """
        Refine the magnification method boundaries using model comparisons.

        For each of the four transition points in ``self.mag_methods``,
        compares the two adjacent magnification methods at the current
        boundary using a magnification-dependent threshold (see
        :meth:`_mag_threshold`). If they differ by more than the threshold,
        uses exponential search and ``scipy.optimize.brentq`` (via
        :meth:`_find_method_boundary`) to locate the outermost time at
        which the methods agree.

        If the refined boundaries are not strictly monotonic (i.e. adjacent
        method windows have collapsed into each other), refinement is
        retried with progressively looser thresholds. The sequence of base
        values tried is ``[base, 0.001, 0.005, 0.01]``, with any values
        smaller than ``base`` skipped. If no threshold produces monotonic
        boundaries, refinement is abandoned and the original unrefined
        boundaries from :meth:`set_mag_method` are restored, and a
        :class:`UserWarning` is issued.

        The four boundaries refined are, from left to right:

        - ``point_source`` / ``hexadecapole`` (left)
        - ``hexadecapole`` / ``VBBL`` (left)
        - ``VBBL`` / ``hexadecapole`` (right)
        - ``hexadecapole`` / ``point_source`` (right)

        Must be called after :meth:`set_mag_method`.

        Parameters
        ----------
        base : float, optional
            Starting base precision level. The first entry in the fallback
            sequence; subsequent values are fixed at ``0.001``, ``0.005``,
            and ``0.01``. Default 0.0001.
        xtol : float, optional
            Time precision in days for each refined boundary.
            Default 0.01.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If :meth:`set_mag_method` has not been called first.
        """
        if self.mag_methods is None or self.params is None:
            raise RuntimeError(
                "set_mag_method() must be called before refine_mag_methods().")

        initial_mag_methods = self.mag_methods.copy()

        model_vbbl = self._make_model('VBBL')
        model_hexa = self._make_model('hexadecapole')
        model_ps = self._make_model('point_source')

        # Try progressively looser thresholds; skip any tighter than base
        fallback_sequence = [b for b in [0.0001, 0.001, 0.005, 0.01]
                             if b >= base]

        for current_base in fallback_sequence:
            self.mag_methods = initial_mag_methods.copy()
            self._apply_refinement(
                current_base, xtol, model_vbbl, model_hexa, model_ps)
            if self._boundaries_monotonic():
                return
            warnings.warn(
                f"Refined mag_method boundaries are not monotonic with "
                f"base={current_base:.4f}. "
                f"Trying next threshold in fallback sequence.",
                UserWarning)

        # All thresholds exhausted: restore original unrefined boundaries
        warnings.warn(
            f"Refined mag_method boundaries are not monotonic for any "
            f"threshold in {fallback_sequence}. "
            f"Falling back to unrefined boundaries.",
            UserWarning)
        self.mag_methods = initial_mag_methods


def get_wide_params(params, limit='GG97'):
    """
    Transform initial anomaly parameters into wide binary lens model parameters.

    Wrapper for :class:`WidePlanetParameterEstimator`.

    Parameters
    ----------
    params : dict
        Anomaly light curve parameters as returned by
        :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
        Required keys:

        - ``'t_0'`` : float, time of maximum magnification.
        - ``'u_0'`` : float, impact parameter.
        - ``'t_E'`` : float, Einstein crossing time.
        - ``'t_pl'`` : float, time of the anomaly.
        - ``'dt'`` : float, duration of the anomaly.
        - ``'dmag'`` : float, magnitude difference of the perturbation.

    limit : str, optional
        Method to use for estimating ``rho``. One of ``'GG97'`` (default),
        ``'dwarf'``, ``'giant'``, or ``'point'``.

    Returns
    -------
    :class:`BinaryLensParams`
        Wide binary lens model parameters.
    """
    estimator = WidePlanetParameterEstimator(params, limit=limit)

    return estimator.binary_params


def get_possible_bump_anomaly_solutions(params):
    """
    Get possible binary lens and binary source solutions for a bump-type anomaly.

    Runs simple, analytic parameter estimators for multiple model types and
    returns all solutions in a single dictionary.

    Parameters
    ----------
    params : dict
        Anomaly light curve parameters as returned by
        :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
        Required keys:

        - ``'t_0'`` : float, time of maximum magnification.
        - ``'u_0'`` : float, impact parameter.
        - ``'t_E'`` : float, Einstein crossing time.
        - ``'t_pl'`` : float, time of the anomaly.
        - ``'dt'`` : float, duration of the anomaly.
        - ``'dmag'`` : float, magnitude difference of the perturbation.

    Returns
    -------
    dict
        Keys are solution types (``'Wide GG97'``, ``'Wide dwarf'``,
        ``'Wide giant'``, ``'CloseUpper'``, ``'CloseLower'``,
        ``'BinarySource'``), values are :class:`BinaryLensParams` or
        :class:`BinarySourceParams` objects.
    """
    solutions = {}

    for limit in ['GG97', 'dwarf', 'giant']:
        estimator = WidePlanetParameterEstimator(params, limit=limit)
        solutions[f'Wide {limit}'] = estimator.get_binary_ulens_params()

    close_upper = CloseUpperBinaryParameterEstimator(params)
    solutions['CloseUpper'] = close_upper.get_binary_lens_params()

    close_lower = CloseLowerBinaryParameterEstimator(params)
    solutions['CloseLower'] = close_lower.get_binary_lens_params()

    solutions['BinarySource'] = get_binary_source_params(params)

    return solutions


class ParameterEstimator():
    """
    Base class for analytic microlensing parameter estimators.

    Provides common properties and methods for computing binary lens
    parameters from initial PSPL parameters and anomaly properties.
    Subclasses should implement :meth:`get_binary_lens_params` and
    optionally override :meth:`get_rho`.

    Parameters
    ----------
    params : dict
        Anomaly light curve parameters as returned by
        :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
        Required keys:

        - ``'t_0'`` : float, time of maximum magnification.
        - ``'u_0'`` : float, impact parameter.
        - ``'t_E'`` : float, Einstein crossing time.
        - ``'t_pl'`` : float, time of the anomaly.

    limit : str, optional
        Method to use for estimating ``rho``. One of ``'dwarf'``,
        ``'giant'``, or ``'point'``.
    """

    def __init__(self, params, limit=None):
        self.params = params
        self.limit = limit

        self._tau_pl, self._u_pl = None, None
        self._s, self._alpha = None, None
        self._q = None
        self._rho = None
        self._binary_params = None

    def get_binary_lens_params(self):
        """
        Return binary lens parameters for this estimator.

        To be implemented by subclasses.

        Returns
        -------
        :class:`BinaryLensParams`
            Binary lens model parameters.
        """
        pass

    def get_rho(self):
        """
        Return rho based on the assumed source size limit.

        Returns a fixed value of rho based on ``self.limit``:

        - ``'dwarf'`` : 0.001
        - ``'giant'`` : 0.05
        - ``'point'`` : None (point source)

        Overridden by subclasses to support additional limits (e.g. ``'GG97'``).

        Returns
        -------
        float or None
            Source size relative to the Einstein radius, or None for a
            point source.

        Raises
        ------
        ValueError
            If ``self.limit`` is not a recognized value.
        """
        if self.limit == 'dwarf':
            return 0.001
        elif self.limit == 'giant':
            return 0.05
        elif self.limit == 'point':
            return None
        else:
            raise ValueError('Your limit for calculating rho is not implemented: ', self.limit)

    @property
    def binary_params(self):
        """
        Binary lens parameters for this estimator.

        Computed lazily on first access via :meth:`get_binary_lens_params`.

        Returns
        -------
        :class:`BinaryLensParams`
            Binary lens model parameters.
        """
        if self._binary_params is None:
            self._binary_params = self.get_binary_lens_params()

        return self._binary_params

    @property
    def t_0(self):
        """Time of maximum magnification, from ``self.params``."""
        return self.params['t_0']

    @property
    def u_0(self):
        """Impact parameter, from ``self.params``."""
        return self.params['u_0']

    @property
    def t_E(self):
        """Einstein crossing time, from ``self.params``."""
        return self.params['t_E']

    @property
    def tau_pl(self):
        """
        Dimensionless time of the anomaly.

        Computed as ``(t_pl - t_0) / t_E``.

        Returns
        -------
        float
        """
        if self._tau_pl is None:
            self._tau_pl = (self.params['t_pl'] - self.params['t_0']) / self.params['t_E']

        return self._tau_pl

    @property
    def u_pl(self):
        """
        Lens-source separation at the anomaly time in Einstein radius units.

        Computed as ``sqrt(u_0^2 + tau_pl^2)``.

        Returns
        -------
        float
        """
        if self._u_pl is None:
            self._u_pl = np.sqrt(self.params['u_0'] ** 2 + self.tau_pl ** 2)

        return self._u_pl

    def _correct_alpha(self, alpha):
        """
        Normalize alpha to the range (-360, 360] degrees.

        Parameters
        ----------
        alpha : float
            Angle in degrees.

        Returns
        -------
        float
            Normalized angle in degrees.
        """
        while alpha > 360.:
            alpha -= 360.

        while alpha < -360:
            alpha += 360.

        return alpha

    @property
    def alpha(self):
        """
        Trajectory angle of the source relative to the lens axis, in degrees.

        Computed from ``u_0`` and ``tau_pl`` using the PSPL geometry.

        Returns
        -------
        float
        """
        if self._alpha is None:
            alpha = np.pi - np.arctan2(self.params['u_0'], self.tau_pl)
            alpha = np.rad2deg(alpha)
            self._alpha = self._correct_alpha(alpha)

        return self._alpha

    @property
    def rho(self):
        """
        Source size relative to the Einstein radius.

        Computed lazily on first access via :meth:`get_rho`. Can be
        overridden by setting directly.

        Returns
        -------
        float or None
            Rho value, or None for a point source.
        """
        if self._rho is None:
            self._rho = self.get_rho()

        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = value


class WidePlanetParameterEstimator(ParameterEstimator):
    """
    Analytic parameter estimator for wide binary lens models.

    Extends :class:`ParameterEstimator` to compute binary lens parameters
    appropriate for a wide planet (s > 1) from anomaly light curve properties.
    Implements the ``'GG97'`` limit for estimating rho in addition to the
    limits defined in the base class.

    Parameters
    ----------
    params : dict
        Anomaly light curve parameters as returned by
        :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
    limit : str, optional
        Method to use for estimating ``rho``. One of ``'GG97'`` (default),
        ``'dwarf'``, ``'giant'``, or ``'point'``.
    """

    def __init__(self, params, limit='GG97'):
        super().__init__(params, limit=limit)
        self._delta_A = None
        self._a_pspl = None

    def get_rho(self):
        """
        Return rho based on the assumed source size limit.

        Extends :meth:`ParameterEstimator.get_rho` to support the ``'GG97'``
        limit, where rho is estimated from the anomaly duration as
        ``dt / t_E / 4``. All other limits fall back to the base class.

        Returns
        -------
        float or None
            Source size relative to the Einstein radius, or None for a
            point source.
        """
        if self.limit == 'GG97':
            rho = self.params['dt'] / self.params['t_E'] / 4.
        else:
            rho = super().get_rho()

        return rho

    def calc_binary_ulens_params(self):
        """
        Compute the binary lens parameter dictionary.

        Assembles the PSPL parameters with the computed binary lens
        parameters (``s``, ``alpha``, ``q``, and optionally ``rho``)
        into a single dictionary suitable for passing to
        ``MulensModel.Model``.

        Returns
        -------
        dict
            Binary lens parameter dictionary with keys ``'t_0'``,
            ``'u_0'``, ``'t_E'``, ``'s'``, ``'alpha'``, and ``'q'``.
            ``'rho'`` is included unless ``limit='point'``.
        """
        new_params = {'t_0': self.t_0, 'u_0': self.u_0, 't_E': self.t_E, 's': self.s, 'alpha': self.alpha}
        rho = self.rho
        if rho is not None:
            new_params['rho'] = rho

        new_params['q'] = self.q

        return new_params

    def get_binary_lens_params(self):
        """
        Return binary lens parameters for the wide planet model.

        Calls :meth:`calc_binary_ulens_params` to get the parameter
        dictionary, wraps it in a :class:`BinaryLensParams` object, and
        sets the magnification methods via
        :meth:`BinaryLensParams.set_mag_method`.

        Returns
        -------
        :class:`BinaryLensParams`
            Wide binary lens model parameters with magnification methods set.
        """
        binary_ulens_params = self.calc_binary_ulens_params()
        out = BinaryLensParams(binary_ulens_params)
        out.set_mag_method(self.params)
        out.refine_mag_methods()
        return out

    @property
    def s(self):
        """
        Binary lens separation in Einstein radius units.

        Computed from ``u_pl`` as ``0.5 * (sqrt(u_pl^2 + 4) + u_pl)``.
        This is the wide-topology solution (s > 1).

        Returns
        -------
        float
        """
        if self._s is None:
            u = self.u_pl
            self._s = 0.5 * (np.sqrt(u ** 2 + 4) + u)
        return self._s

    @property
    def q(self):
        """
        Planet-to-star mass ratio.

        Estimated from the anomaly as ``0.5 * |delta_A| * rho^2``.

        Returns
        -------
        float
        """
        if self._q is None:
            self._q = 0.5 * np.abs(self.delta_A) * (self.rho ** 2)

        return self._q

    @property
    def a_pspl(self):
        """
        PSPL magnification at the anomaly position.

        Computed from ``u_pl`` using the standard PSPL formula:
        ``(u_pl^2 + 2) / sqrt(u_pl^2 * (u_pl^2 + 4))``.

        Returns
        -------
        float
        """
        if self._a_pspl is None:
            self._a_pspl = (self.u_pl ** 2 + 2.) / np.sqrt(self.u_pl ** 2 * (self.u_pl ** 2 + 4.))

        return self._a_pspl

    @property
    def delta_A(self):
        """
        Change in magnification due to the planetary anomaly.

        Computed as ``a_pspl * (10^(dmag / -2.5) - 1)``. This assumes
        zero blending (``fb = 0``), which may be inaccurate for events
        with significant blend flux.

        Returns
        -------
        float
        """
        # TODO: Might want to add an option to calculate delta_A using PSPL fitted fs and fb.
        # Current calculation assumes fb=0. This could be a problem if fb is large, e.g. OB180383.
        if self._delta_A is None:
            self._delta_A = self.a_pspl * (10. ** (self.params['dmag'] / -2.5) - 1.)

        return self._delta_A


class WidePlanetGridSearchEstimator(WidePlanetParameterEstimator):
    """
    Estimates wide planet binary lens parameters by performing a chi2 grid
    search centered on the analytic parameter estimates from
    WidePlanetParameterEstimator.

    The grid spans alpha, s, log_q, and log_rho. The best-fit parameters
    are identified by minimizing chi2 over the grid.

    Parameters
    ----------
    datasets : list of MulensModel.MulensData
        Photometric datasets to evaluate chi2 against.
    params : dict
        Anomaly light curve parameters as returned by
        :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
    model_config : ModelConfig, optional
        Configuration for Model construction. If None, a default
        ``ModelConfig`` is used (no coords, no limb darkening).
    event_config : EventConfig, optional
        Configuration for Event construction. If None, a default
        ``EventConfig`` is used (no coords, no flux fixing).
    d_alpha : float, optional
        Step size for alpha grid. See :attr:`alpha_values`.
    n_alpha : int, optional
        Number of alpha grid points. See :attr:`alpha_values`.
    d_s : float, optional
        Step size for s grid. See :attr:`s_values`.
    n_s : int, optional
        Number of s grid points. See :attr:`s_values`.
    log_q_values : array-like, optional
        Grid values for log10(q). See :attr:`log_q_grid`.
    log_rho_values : array-like, optional
        Grid values for log10(rho). See :attr:`log_rho_grid`.
    alpha_grid : array-like, optional
        Explicit grid values for alpha. If provided, overrides d_alpha
        and n_alpha. Defaults to None.
    s_grid : array-like, optional
        Explicit grid values for s. If provided, overrides d_s and n_s.
        Defaults to None.
    refine : bool, optional
        If True, runs Nelder-Mead refinement after the grid search.
        Defaults to True.
    nelder_mead_options : dict, optional
        Options passed to scipy.optimize.minimize with method='Nelder-Mead'.
        Supported keys: 'maxfev' (default 500), 'xatol' (default 1e-3),
        'fatol' (default 0.1). Any key not specified falls back to the
        default. Note: 'initial_simplex' is computed internally from the
        grid step sizes and should not be passed here.

    Note
    ----
    In future it might be a good idea to refactor best_params (and
    related methods) to use dynamic lists of grid parameters rather than
    hardcoding ['alpha', 's', 'q', 'rho'].
    """

    def __init__(self, datasets, params, model_config=None, event_config=None,
                 d_alpha=None, n_alpha=None,
                 d_s=None, n_s=None,
                 log_q_values=None, log_rho_values=None,
                 alpha_grid=None, s_grid=None,
                 refine=True,
                 nelder_mead_options=None):
        super().__init__(params)
        self.datasets = datasets
        self.model_config = model_config if model_config is not None else ModelConfig()
        self.event_config = event_config if event_config is not None else EventConfig()

        self.d_alpha = d_alpha
        self.n_alpha = n_alpha
        self.d_s = d_s
        self.n_s = n_s
        self.log_q_values = log_q_values
        self.log_rho_values = log_rho_values
        self._alpha_grid = alpha_grid
        self._s_grid = s_grid
        self.refine = refine
        self.nelder_mead_options = nelder_mead_options
        self._results = None
        self._refinement_results = None
        self._refinement_result = None
        self._all_results = None
        self._is_run = False

    @property
    def _base_binary_params(self):
        """
        Internal access to binary_params without run check. Used by all
        internal methods to avoid triggering the RuntimeError guard on
        binary_params before run() has been called.
        """
        if self._binary_params is None:
            self._binary_params = self.get_binary_lens_params()
        return self._binary_params

    @property
    def _nelder_mead_options(self):
        """
        Nelder-Mead options with defaults applied.

        Merges user-supplied ``nelder_mead_options`` with defaults
        (``maxfev=500``, ``xatol=1e-3``, ``fatol=0.1``). User-supplied
        values take precedence.

        Returns
        -------
        dict
            Options dict suitable for passing to
            ``scipy.optimize.minimize``.
        """
        defaults = {'maxfev': 500, 'xatol': 1e-3, 'fatol': 0.1}
        if self.nelder_mead_options is not None:
            defaults.update(self.nelder_mead_options)
        return defaults

    @property
    def binary_params(self):
        """
        Best-fit binary lens parameters from the grid search and refinement.

        :meth:`run` must be called before accessing this property.

        Returns
        -------
        :class:`BinaryLensParams`
            Binary lens parameters populated with best-fit values.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called.
        """
        if not self._is_run:
            raise RuntimeError(
                "binary_params is not available until run() has been called.")
        return self._binary_params

    @property
    def alternate_params(self):
        """
        Degenerate binary lens parameters using the s_dagger degeneracy.

        Computes the alternate solution by replacing ``s`` with
        ``s_analytic^2 / s_best``, where ``s_analytic`` is the analytic
        wide-planet estimate and ``s_best`` is the best-fit value from the
        grid search.

        see Hwang et al. 2022 and Ryu et al. 2022 for more background

        https://ui.adsabs.harvard.edu/abs/2022AJ....163...43H/abstract
        https://ui.adsabs.harvard.edu/abs/2022AJ....164..180R/abstract

        Returns
        -------
        :class:`BinaryLensParams`
            Alternate binary lens parameters.
        """
        base_params = self.get_binary_lens_params()
        s_new = base_params.ulens['s'] ** 2 / self.best_params['s']
        alt_params = BinaryLensParams(base_params.ulens)
        alt_params.mag_methods = base_params.mag_methods
        alt_params.ulens['s'] = s_new
        return alt_params

    @property
    def best_params(self):
        """
        Best-fit parameter dictionary from the grid search and refinement.

        :meth:`run` must be called before accessing this property.

        Returns
        -------
        dict
            Binary lens parameter dictionary.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called.
        """
        if not self._is_run:
            raise RuntimeError(
                "best_params is not available until run() has been called.")
        return self._binary_params.ulens

    def run(self):
        """
        Run the full pipeline: grid search and optional Nelder-Mead refinement.

        Populates :attr:`binary_params` and :attr:`best_params`. Must be
        called before accessing those properties.

        Returns
        -------
        :class:`BinaryLensParams`
            Binary lens parameters populated with best-fit values.
        """
        _ = self.all_results  # triggers grid search + refinement
        self._is_run = True
        return self._binary_params

    @property
    def alpha_values(self):
        """
        Grid values of alpha for the grid search.

        If ``alpha_grid`` was provided at construction, returns that
        directly. Otherwise, constructs a uniform grid of ``n_alpha=6``
        points centered on the analytic ``alpha`` estimate with step
        size ``d_alpha=0.1``.

        Returns
        -------
        numpy.ndarray
        """
        if self._alpha_grid is not None:
            return self._alpha_grid
        d_alpha = self.d_alpha if self.d_alpha is not None else 0.1
        n_alpha = self.n_alpha if self.n_alpha is not None else 6
        alpha_offset = np.arange(n_alpha) - (n_alpha - 1) / 2
        return self.alpha + alpha_offset * d_alpha

    @property
    def s_values(self):
        """
        Grid values of s for the grid search.

        If ``s_grid`` was provided at construction, returns that directly.
        Otherwise, constructs a uniform grid of ``n_s=4`` points centered
        on the analytic ``s`` estimate with step size ``d_s=0.01 * s``.

        Returns
        -------
        numpy.ndarray
        """
        if self._s_grid is not None:
            return self._s_grid
        d_s = self.d_s if self.d_s is not None else 0.01 * self.s
        n_s = self.n_s if self.n_s is not None else 4
        s_offset = np.arange(n_s) - (n_s - 1) / 2
        return self.s + s_offset * d_s

    @property
    def log_q_grid(self):
        """
        Grid values of log10(q) for the grid search.

        Returns ``log_q_values`` if provided at construction, otherwise
        defaults to ``numpy.arange(-6, -1)``.

        Returns
        -------
        numpy.ndarray
        """
        return self.log_q_values if self.log_q_values is not None else np.arange(-6, -1)

    @property
    def log_rho_grid(self):
        """
        Grid values of log10(rho) for the grid search.

        Returns ``log_rho_values`` if provided at construction, otherwise
        defaults to ``numpy.arange(-4, -1)``.

        Returns
        -------
        numpy.ndarray
        """
        return self.log_rho_values if self.log_rho_values is not None else np.arange(-4, -1)

    def _make_event(self, grid_params):
        """
        Create a MulensModel.Event for the given grid parameters.

        Magnification methods and default magnification method are
        model-type specific and are taken from the base binary parameter
        estimates. Coordinates and flux-fixing are applied via
        ``event_config``.

        Parameters
        ----------
        grid_params : dict
            Model parameters for this grid point.

        Returns
        -------
        MulensModel.Event
        """
        model = self.model_config.build(
            parameters=grid_params,
            magnification_methods=self._base_binary_params.mag_methods,
            default_magnification_method='point_source_point_lens',
        )
        return self.event_config.build(
            model=model,
            datasets=self.datasets,
        )

    def _grid_iterator(self):
        """
        Yield all combinations of alpha, s, log_q, and log_rho for the grid search.

        Returns
        -------
        itertools.product
            Iterator over (alpha, s, log_q, log_rho) tuples.
        """
        return product(
            self.alpha_values, self.s_values,
            self.log_q_grid, self.log_rho_grid)

    def _run_grid_search(self):
        """
        Run the chi2 grid search over alpha, s, log_q, and log_rho.

        Iterates over all grid points via :meth:`_grid_iterator`, evaluates
        chi2 for each, and updates ``_base_binary_params`` with the
        best-fit values.

        Returns
        -------
        pandas.DataFrame
            Results for all grid points with columns ``'chi2'``,
            ``'alpha'``, ``'s'``, ``'q'``, ``'rho'``.
        """
        results = []
        grid_params = self._base_binary_params.ulens.copy()

        event = self._make_event(grid_params)

        for alpha, s, log_q, log_rho in self._grid_iterator():
            event.model.parameters.alpha = alpha
            event.model.parameters.s = s
            event.model.parameters.q = 10. ** log_q
            event.model.parameters.rho = 10. ** log_rho

            results.append({
                'chi2': event.get_chi2(),
                'alpha': alpha,
                's': s,
                'q': event.model.parameters.q,
                'rho': event.model.parameters.rho
            })

        df = pd.DataFrame(results)
        best_row = df.loc[df['chi2'].idxmin()]
        self._base_binary_params.ulens.update(best_row[['alpha', 's', 'q', 'rho']].to_dict())
        return df

    def _run_refinement(self):
        """
        Run Nelder-Mead refinement starting from the best grid search point.

        Optimizes over alpha, s, log_q, and log_rho using
        ``scipy.optimize.minimize`` with ``method='Nelder-Mead'``. The
        initial simplex is scaled to the grid step sizes. Updates
        ``_base_binary_params`` only if the refinement finds a lower chi2
        than the grid search best.

        A warning is issued if Nelder-Mead does not converge.

        Returns
        -------
        pandas.DataFrame
            Trajectory of all points evaluated during refinement, with
            columns ``'chi2'``, ``'alpha'``, ``'s'``, ``'q'``, ``'rho'``.
        """
        best = self._base_binary_params.ulens.copy()
        x0 = np.array([
            best['alpha'],
            best['s'],
            np.log10(best['q']),
            np.log10(best['rho'])
        ])

        # Build initial simplex scaled to the grid step sizes used in the
        # grid search. This is important: Nelder-Mead's default simplex
        # perturbs each coordinate by 5% of x0, which is arbitrary and can
        # be badly scaled here (e.g. log_q near 0 gets almost no perturbation).
        d_alpha = self.d_alpha if self.d_alpha is not None else 0.1
        d_s = self.d_s if self.d_s is not None else 0.01 * self.s
        simplex_deltas = np.array([d_alpha, d_s, 0.5, 0.5])
        n = len(x0)
        initial_simplex = np.vstack(
            [x0] + [x0 + simplex_deltas[i] * np.eye(n)[i] for i in range(n)])

        # Single Event created once; parameters updated in-place each call
        event = self._make_event(best)

        trajectory = []

        def chi2_fn(x):
            alpha, s, log_q, log_rho = x
            event.model.parameters.alpha = alpha
            event.model.parameters.s = s
            event.model.parameters.q = 10. ** log_q
            event.model.parameters.rho = 10. ** log_rho
            chi2 = event.get_chi2()
            trajectory.append({
                'chi2': chi2,
                'alpha': alpha,
                's': s,
                'q': 10. ** log_q,
                'rho': 10. ** log_rho,
            })
            return chi2

        try:
            result = minimize(
                chi2_fn, x0, method='Nelder-Mead',
                options={**self._nelder_mead_options, 'initial_simplex': initial_simplex})
        except Exception as e:
            warnings.warn(
                f"Nelder-Mead refinement exited in an error. Error:\n{type(e).__name__}: {e}.")
            return None

        if not result.success:
            warnings.warn(
                f"Nelder-Mead refinement did not converge: {result.message}. "
                f"Best chi2={result.fun:.4f} after {result.nfev} evaluations.")

        self._refinement_result = result

        df = pd.DataFrame(trajectory)
        # Guard against Nelder-Mead wandering to a worse basin than the grid:
        # take the global best across both grid and refinement trajectory.
        # Use result.x directly — scipy guarantees this is the best point found
        best_grid_chi2 = self.results['chi2'].min()
        if result.fun < best_grid_chi2:
            alpha, s, log_q, log_rho = result.x
            self._base_binary_params.ulens.update({
                'alpha': alpha,
                's': s,
                'q': 10. ** log_q,
                'rho': 10. ** log_rho
            })
        # else: grid best is already set by _run_grid_search — leave it

        return df

    @property
    def results(self):
        """
        Grid search results as a DataFrame.

        Computed lazily on first access by running :meth:`_run_grid_search`
        and :meth:`_postprocess_grid_results`. Columns include ``'chi2'``,
        ``'alpha'``, ``'s'``, ``'q'``, ``'rho'``, ``'log_q'``,
        ``'log_rho'``, and ``'sigma'``.

        Returns
        -------
        pandas.DataFrame
        """
        if self._results is None:
            df = self._run_grid_search()
            self._results = self._postprocess_grid_results(df)
        return self._results

    @property
    def refinement_result(self):
        """Raw scipy OptimizeResult from Nelder-Mead. Check result.success and
        result.nfev for convergence diagnostics."""
        _ = self.refinement_results  # ensure refinement has run
        return self._refinement_result

    @property
    def refinement_results(self):
        """DataFrame of all points evaluated during Nelder-Mead refinement."""
        if self._refinement_results is None:
            _ = self.results  # ensure grid search has run first
            self._refinement_results = self._run_refinement()
        return self._refinement_results

    @property
    def all_results(self):
        """
        Combined grid search and refinement results as a DataFrame.

        Computed lazily on first access. Merges :attr:`results` and (if
        ``refine=True``) :attr:`refinement_results`, adding a ``'source'``
        column (``'grid'`` or ``'refinement'``) and recomputing ``'sigma'``
        relative to the global minimum chi2.

        Returns
        -------
        pandas.DataFrame
            Columns include ``'chi2'``, ``'alpha'``, ``'s'``, ``'q'``,
            ``'rho'``, ``'log_q'``, ``'log_rho'``, ``'sigma'``,
            and ``'source'``.
        """
        if self._all_results is None:
            df_grid = self.results.copy()
            df_grid['source'] = 'grid'
            df_grid['iteration'] = 0

            if self.refine and self.refinement_results is not None:
                df_refine = self.refinement_results.copy()
                df_refine['source'] = 'refinement'
                df_refine['log_q'] = np.round(np.log10(df_refine['q'])).astype(int)
                df_refine['log_rho'] = np.round(np.log10(df_refine['rho'])).astype(int)
                combined = pd.concat([df_grid, df_refine], ignore_index=True)
            else:
                combined = df_grid

            # Recompute sigma relative to global minimum
            min_chi2 = combined['chi2'].min()
            combined['sigma'] = np.sqrt(combined['chi2'] - min_chi2)
            self._all_results = combined

        return self._all_results

    def _postprocess_grid_results(self, df):
        """
        Add log_q, log_rho, and sigma columns to the raw grid results.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw grid results with columns ``'chi2'``, ``'alpha'``, ``'s'``,
            ``'q'``, ``'rho'``.

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with added columns ``'log_q'``, ``'log_rho'``,
            and ``'sigma'`` (relative to the minimum chi2 in this DataFrame).
        """
        df = df.copy()
        df['log_q'] = np.round(np.log10(df['q'])).astype(int)
        df['log_rho'] = np.round(np.log10(df['rho'])).astype(int)
        df['sigma'] = np.sqrt(df['chi2'] - df['chi2'].min())
        return df

    def get_results_within_n_sigma(self, n_sigma=3):
        """
        Return all results within n_sigma of the minimum chi2.

        Parameters
        ----------
        n_sigma : float, optional
            Maximum sigma threshold. Default is 3.

        Returns
        -------
        pandas.DataFrame
            Subset of :attr:`all_results` with ``sigma <= n_sigma``.
        """
        df = self.all_results
        return df[df['sigma'] <= n_sigma]

    @staticmethod
    def _get_sigma_marker(sigma):
        """
        Return a matplotlib marker style and size for a given sigma value.

        Used by :meth:`plot_sigma_maps` to distinguish refinement points
        by their sigma level.

        Parameters
        ----------
        sigma : float
            Sigma value to classify.

        Returns
        -------
        marker : str
            Matplotlib marker code (``'*'``, ``'D'``, ``'o'``, or ``'^'``).
        size : int
            Marker size in points.
        """
        if sigma < 1:
            return '*', 200
        elif sigma < 2:
            return 'D', 100
        elif sigma < 3:
            return 'o', 60
        else:
            return '^', 30

    def plot_sigma_maps(self):
        """
        Plot 2D sigma maps in the alpha-s plane for each log_q and log_rho combination.

        Produces one figure per unique log_q value. Each figure contains
        one subplot per unique log_rho value, showing a heatmap of sigma
        (relative to the global minimum chi2) over the alpha-s grid. If
        ``refine=True``, refinement trajectory points are overlaid as
        scatter points with marker styles from :meth:`_get_sigma_marker`.

        Note
        ----
        Refinement points are only overlaid when their rounded log_q and
        log_rho values match the grid values exactly. Refinement points
        that have wandered to different log_q or log_rho values will not
        be shown. This is a known limitation and may be addressed in a
        future version.
        """
        df_all = self.all_results
        df_grid = df_all[df_all['source'] == 'grid']

        unique_log_q = sorted(df_grid['log_q'].unique())
        unique_log_rho = sorted(df_grid['log_rho'].unique())
        n_rho = len(unique_log_rho)

        # TODO: Refinement points are only overlaid when their rounded log_q and
        # log_rho match the grid values exactly. Refinement points that have
        # wandered to different log_q or log_rho values will not be shown. Fix this in the future.
        if self.refine:
            df_refine = df_all[df_all['source'] == 'refinement']

        for log_q in unique_log_q:
            fig = plt.figure(figsize=(10, 4 * n_rho))
            gs = GridSpec(n_rho, 1, figure=fig, hspace=0.3)

            for idx, log_rho in enumerate(unique_log_rho):
                ax = fig.add_subplot(gs[idx, 0])

                # Grid imshow
                mask = (df_grid['log_q'] == log_q) & (df_grid['log_rho'] == log_rho)
                subset = df_grid[mask]
                grid = subset.pivot(index='s', columns='alpha', values='sigma')
                im = ax.imshow(grid, cmap='Set1', vmin=0, vmax=100, aspect='auto',
                               origin='lower',
                               extent=[subset['alpha'].min(), subset['alpha'].max(),
                                       subset['s'].min(), subset['s'].max()])

                # Refinement scatter overlay
                if self.refine:
                    refine_mask = (
                            (df_refine['log_q'] == log_q) &
                            (df_refine['log_rho'] == log_rho))
                    refine_subset = df_refine[refine_mask]

                    for sigma_low, sigma_high in [(0, 1), (1, 2), (2, 3), (3, np.inf)]:
                        pts = refine_subset[
                            (refine_subset['sigma'] >= sigma_low) &
                            (refine_subset['sigma'] < sigma_high)]
                        if not pts.empty:
                            marker, size = self._get_sigma_marker(sigma_low)
                            ax.scatter(pts['alpha'], pts['s'],
                                       marker=marker, s=size,
                                       c=pts['sigma'], cmap='Set1', vmin=0, vmax=100,
                                       edgecolors='black', linewidths=0.5, zorder=5)

                ax.set_xlabel('alpha', fontsize=10)
                ax.set_ylabel('s', fontsize=10)
                ax.set_title(f'log_q={log_q}, log_rho={log_rho}', fontsize=11)

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('sigma', fontsize=10)

            fig.suptitle(f'log_q = {log_q}', fontsize=13, fontweight='bold')
            plt.tight_layout()


class WidePlanetEnsembleInitializer():
    """
    Builds an ensemble of starting points for emcee by running multiple
    WidePlanetGridSearchEstimators with perturbed PSPL parameters.

    The first estimator uses a broad default grid. Its best log_q and
    log_rho are used to seed a narrower grid for all subsequent estimators.

    Parameters
    ----------
    datasets : list of MulensModel.MulensData
        Photometric datasets.
    anomaly_params : dict
        Anomaly light curve parameters as returned by
        :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
    sigmas : dict
        Step sizes for PSPL parameter perturbations. Expected keys:
        ``'t_0'``, ``'u_0'``, ``'t_E'``. See :attr:`sigma_t0`,
        :attr:`sigma_u0`, :attr:`sigma_tE` for defaults.
    model_config : ModelConfig, optional
        Configuration for Model construction. If None, a default
        ``ModelConfig`` is used (no coords, no limb darkening).
    event_config : EventConfig, optional
        Configuration for Event construction. If None, a default
        ``EventConfig`` is used (no coords, no flux fixing).
    n_estimators : int, optional
        Number of estimators to run. Should equal n_walkers. Default is 40.
    pspl_chi2 : float, optional
        Chi2 of the no-planet PSPL model. Used only for diagnostics
        (delta_chi2, summary counts). Default is None.
    """
    # TODO: Hypothesis that this is very slow because the event/Estimator class is getting created anew every time.

    def __init__(self, datasets, anomaly_params, sigmas, model_config=None,
                 event_config=None, n_estimators=40, pspl_chi2=None):
        self.datasets = datasets
        self.anomaly_params = anomaly_params
        self.sigmas = sigmas
        self.model_config = model_config if model_config is not None else ModelConfig()
        self.event_config = event_config if event_config is not None else EventConfig()

        self.n_estimators = n_estimators
        self.pspl_chi2 = pspl_chi2

        self._results = None
        self._mag_methods = None
        self._initial_model = None
        self._seed_log_q = None
        self._seed_log_rho = None

    @property
    def sigma_t0(self):
        """Perturbation step size for t_0, from ``self.sigmas`` (default 0.00001)."""
        return self.sigmas.get('t_0', 0.00001)

    @property
    def sigma_u0(self):
        """Perturbation step size for u_0, from ``self.sigmas`` (default 0.001 * u_0)."""
        return self.sigmas.get('u_0', 0.001 * self.anomaly_params['u_0'])

    @property
    def sigma_tE(self):
        """Perturbation step size for t_E, from ``self.sigmas`` (default 0.001 * t_E)."""
        return self.sigmas.get('t_E', 0.001 * self.anomaly_params['t_E'])

    def _perturb_params(self):
        """
        Generate one set of perturbed PSPL parameters.

        Override to implement different perturbation strategies.

        Returns
        -------
        dict
            Perturbed anomaly_params.
        """
        params = self.anomaly_params.copy()
        params['t_0'] = self.anomaly_params['t_0'] + np.random.randn() * self.sigma_t0
        params['u_0'] = self.anomaly_params['u_0'] + np.random.randn() * self.sigma_u0
        params['t_E'] = self.anomaly_params['t_E'] + np.random.randn() * self.sigma_tE
        return params

    def _get_seeded_grid_values(self, best_log_val):
        """
        Generate a 3-point grid from the seed estimator's best log value.

        Perturbs ``best_log_val`` by 5% and returns
        ``[rand_best - 0.5, rand_best, rand_best + 0.5]``.

        Parameters
        ----------
        best_log_val : float
            Best log10 value from the seed estimator.

        Returns
        -------
        list of float
            Three grid values centered on the perturbed best log value.
        """
        rand_best = best_log_val + np.random.randn() * 0.05 * np.abs(best_log_val)
        return [rand_best - 0.5, rand_best, rand_best + 0.5]

    def _run_single_estimator(self, params, log_q_values=None, log_rho_values=None):
        """
        Run a single WidePlanetGridSearchEstimator for the given params.

        Override to use different estimator settings.

        Parameters
        ----------
        params : dict
            Anomaly parameters for this estimator.
        log_q_values : list, optional
            If provided, passed as the log_q grid. If None, the
            estimator uses its default broad grid.
        log_rho_values : list, optional
            If provided, passed as the log_rho grid. If None, the
            estimator uses its default broad grid.

        Returns
        -------
        best : dict
            Best-fit binary lens parameters.
        mag_methods : list
            Magnification methods from this estimator.
        """
        estimator = WidePlanetGridSearchEstimator(
            datasets=self.datasets,
            params=params,
            model_config=self.model_config,
            event_config=self.event_config,
            refine=True,
            log_q_values=log_q_values,
            log_rho_values=log_rho_values,
        )
        estimator.run()
        return estimator.binary_params.ulens.copy(), estimator.binary_params.mag_methods

    def _evaluate_chi2(self, best, mag_methods):
        """
        Compute chi2 for a set of binary lens parameters.

        Parameters
        ----------
        best : dict
            Binary lens parameter dictionary.
        mag_methods : list
            Magnification methods in MulensModel convention.

        Returns
        -------
        float
            Chi2 value for the given parameters.
        """
        model = self.model_config.build(
            parameters=best,
            magnification_methods=mag_methods,
            default_magnification_method='point_source_point_lens',
        )
        event = self.event_config.build(
            model=model,
            datasets=self.datasets,
        )
        return event.get_chi2()

    def _run_all_estimators(self):
        """
        Run all n_estimators and collect results into a DataFrame.

        The first estimator uses the default broad grid. Its best log_q
        and log_rho seed all subsequent estimators via
        _get_seeded_grid_values().
        """
        rows = []

        for i in range(self.n_estimators):
            params = self._perturb_params()

            if self._seed_log_q is None:
                best, mag_methods = self._run_single_estimator(params)
                self._seed_log_q = np.log10(best['q'])
                self._seed_log_rho = np.log10(best['rho'])
            else:
                log_q_values = self._get_seeded_grid_values(self._seed_log_q)
                log_rho_values = self._get_seeded_grid_values(self._seed_log_rho)
                best, mag_methods = self._run_single_estimator(
                    params,
                    log_q_values=log_q_values,
                    log_rho_values=log_rho_values)

            if self._mag_methods is None:
                self._mag_methods = mag_methods

            chi2 = self._evaluate_chi2(best, mag_methods)

            row = {
                'chi2': chi2,
                't_0': best['t_0'],
                'u_0': best['u_0'],
                't_E': best['t_E'],
                's': best['s'],
                'q': best['q'],
                'rho': best['rho'],
                'alpha': best['alpha']
            }
            if self.pspl_chi2 is not None:
                row['delta_chi2'] = self.pspl_chi2 - chi2

            rows.append(row)

            log_str = (f'Estimator {i:3d}: chi2={chi2:.2f}  '
                       f't_E={best["t_E"]:.3f}  '
                       f'log_q={np.log10(best["q"]):.2f}  '
                       f'log_rho={np.log10(best["rho"]):.2f}  '
                       f'{"[seed]" if i == 0 else "[seeded]"}')
            if self.pspl_chi2 is not None:
                log_str += f'  delta_chi2={self.pspl_chi2 - chi2:.2f}'
            print(log_str)

        return pd.DataFrame(rows)

    @property
    def results(self):
        """
        Results from every estimator as a DataFrame.

        Computed lazily on first access by running :meth:`_run_all_estimators`.
        Columns include ``'chi2'``, ``'t_0'``, ``'u_0'``, ``'t_E'``, ``'s'``,
        ``'q'``, ``'rho'``, ``'alpha'``, and (if ``pspl_chi2`` was provided)
        ``'delta_chi2'``.

        Returns
        -------
        pandas.DataFrame
        """
        if self._results is None:
            self._results = self._run_all_estimators()
        return self._results

    @property
    def mag_methods(self):
        """
        Magnification methods from the seed estimator.

        Computed lazily as a side effect of accessing :attr:`results`.
        Taken from the first estimator run (the seed estimator with the
        broad default grid).

        Returns
        -------
        list
            Magnification methods in MulensModel convention.
        """
        _ = self.results  # ensure estimators have run
        return self._mag_methods

    @property
    def initial_model(self):
        """
        Best-fit binary lens parameters across all estimators (lowest chi2).
        """
        if self._initial_model is None:
            df = self.results
            best_row = df.loc[df['chi2'].idxmin()]
            self._initial_model = {
                k: best_row[k]
                for k in ['t_0', 'u_0', 't_E', 's', 'q', 'rho', 'alpha']}
        return self._initial_model

    def summary(self):
        """
        Print a summary of all estimator results sorted by chi2.
        """
        df = self.results
        if 'delta_chi2' in df.columns:
            n_better = np.sum(df['delta_chi2'] > 0)
            print(f'\n{n_better} / {self.n_estimators} estimators better than PSPL')

        cols = ['chi2', 't_E', 'u_0', 's', 'q', 'rho', 'alpha']
        if 'delta_chi2' in df.columns:
            cols.insert(1, 'delta_chi2')
        print('\nSorted by chi2:')
        print(df.sort_values('chi2')[cols].to_string())

    def plot_chi2_distribution(self):
        """
        Plot histogram of chi2 values across all estimators.
        """
        df = self.results
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['chi2'], bins=20)
        if self.pspl_chi2 is not None:
            ax.axvline(self.pspl_chi2, color='red', linestyle='--',
                       label=f'PSPL chi2={self.pspl_chi2:.0f}')
            ax.legend()
        ax.set_xlabel('chi2')
        ax.set_ylabel('N')
        ax.set_title(f'{self.n_estimators} estimators: chi2 distribution')

    def plot_models(self):
        """
        Plot all estimator models in the anomaly region and VBBL zoom,
        colour-coded by chi2 (red=worst, green=best).
        """
        df = self.results
        t_range_anomaly = [self.mag_methods[2], self.mag_methods[8]]
        t_range_vbbl = [self.mag_methods[4], self.mag_methods[6]]

        ref_model = self.model_config.build(
            parameters=self.initial_model,
            magnification_methods=self.mag_methods,
            default_magnification_method='point_source_point_lens',
        )
        ref_event = self.event_config.build(
            model=ref_model,
            datasets=self.datasets,
        )
        source_flux, blend_flux = ref_event.get_ref_fluxes()

        sorted_idx = df['chi2'].argsort().values[::-1]  # worst first
        cmap = plt.cm.get_cmap('RdYlGn', self.n_estimators)

        for fig_title, t_range in [('Anomaly region', t_range_anomaly),
                                   ('VBBL zoom', t_range_vbbl)]:
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.sca(ax)
            ref_event.plot_data()

            for rank, idx in enumerate(sorted_idx):
                row = df.iloc[idx]
                params = {k: row[k] for k in
                          ['t_0', 'u_0', 't_E', 's', 'q', 'rho', 'alpha']}
                model = self.model_config.build(
                    parameters=params,
                    magnification_methods=self.mag_methods,
                    default_magnification_method='point_source_point_lens',
                )
                model.plot_lc(source_flux=source_flux, blend_flux=blend_flux,
                              color=cmap(rank), alpha=0.6, t_range=t_range)

            ref_event.plot_model(label='Best (grid)', color='black',
                                 zorder=10, t_range=t_range, linewidth=2)
            ax.set_xlim(t_range)
            ax.set_xlabel('Time (HJD)')
            ax.set_ylabel('Magnitude')
            ax.set_title(f'{fig_title} — red=worst, green=best chi2')
            ax.minorticks_on()


class CloseUpperBinaryParameterEstimator(WidePlanetParameterEstimator):
    """
    Analytic parameter estimator for close binary lens models (upper caustic).

    Extends :class:`WidePlanetParameterEstimator` to compute binary lens
    parameters appropriate for a close binary (s < 1). The binary lens
    separation uses the close-topology solution, and ``alpha`` is computed
    using the upper caustic geometry.

    Parameters
    ----------
    params : dict
        Anomaly light curve parameters as returned by
        :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
    limit : str, optional
        Method to use for estimating ``rho``. One of ``'dwarf'`` (default),
        ``'giant'``, or ``'point'``. ``'GG97'`` is not supported and will
        raise a ``ValueError``.
    q : float, optional
        Planet-to-star mass ratio. Default is 0.004.
    rho : float, optional
        Source size relative to the Einstein radius. If provided, overrides
        the value computed from ``limit``. Default is None.
    """

    def __init__(self, params, limit='dwarf', q=None, rho=None):
        if limit == 'GG97':
            raise ValueError(
                "'GG97' limit is not supported for close binary models. "
                "Use 'dwarf', 'giant', or 'point'.")
        super().__init__(params, limit=limit)
        if q is None:
            q = 0.004
        self._q = q
        if rho is not None:
            self._rho = rho
        self._eta_not, self._mu, self._phi = None, None, None

    def setup_close_ulens_params(self):
        """
        Assemble the close binary lens parameter dictionary without alpha.

        Builds a parameter dict from the PSPL parameters and computed ``s``,
        ``q``, and (if not None) ``rho``. Used by :meth:`calc_binary_ulens_params`.

        Returns
        -------
        dict
            Binary lens parameter dictionary with keys ``'t_0'``, ``'u_0'``,
            ``'t_E'``, ``'s'``, ``'q'``, and (if ``rho`` is not None) ``'rho'``.
        """
        new_params = {'t_0': self.t_0,
                      'u_0': self.u_0,
                      't_E': self.t_E,
                      's': self.s,
                      'q': self.q}

        if self.rho is not None:
            new_params['rho'] = self.rho

        return new_params

    def calc_binary_ulens_params(self):
        """
        Compute the binary lens parameter dictionary including alpha.

        Calls :meth:`setup_close_ulens_params` and adds the computed
        ``alpha`` for the upper caustic geometry.

        Returns
        -------
        dict
            Binary lens parameter dictionary with keys ``'t_0'``, ``'u_0'``,
            ``'t_E'``, ``'s'``, ``'q'``, ``'alpha'``, and (if ``rho`` is
            not None) ``'rho'``.
        """
        new_params = self.setup_close_ulens_params()
        new_params['alpha'] = self.alpha

        return new_params

    @property
    def log_q_grid(self):
        """
        Grid values of log10(q) for the grid search.

        Returns ``log_q_values`` if provided at construction, otherwise
        defaults to ``numpy.array([-2.5, -2, -1, -0.5])``, which spans
        the mass ratio range appropriate for close binary models.

        Returns
        -------
        numpy.ndarray
        """
        return self.log_q_values if self.log_q_values is not None else np.array([-2.5, -2, -1, -0.5])

    @property
    def s(self):
        """
        Binary lens separation in Einstein radius units.

        Computed from ``u_pl`` as ``0.5 * (sqrt(u_pl^2 + 4) - u_pl)``.
        This is the close-topology solution (s < 1).

        Returns
        -------
        float
        """
        if self._s is None:
            u = self.u_pl
            self._s = 0.5 * (np.sqrt(u ** 2 + 4) - u)

        return self._s

    @property
    def q(self):
        """Planet-to-star mass ratio, set at construction (default 0.004)."""
        return self._q

    @property
    def eta_not(self):
        """
        Vertical distance of the planetary caustic from the binary axis.

        Computed as ``(q^0.5 / s) * (1/sqrt(1 + s^2) + sqrt(1 - s^2))``.
        Used to calculate :attr:`mu`.

        See Han 2006 https://ui.adsabs.harvard.edu/abs/2006ApJ...638.1080H/abstract

        Returns
        -------
        float
        """
        if self._eta_not is None:
            self._eta_not = (self.q ** 0.5 / self.s) * (1 / (np.sqrt(1 + self.s ** 2)) + np.sqrt(1 - self.s ** 2))

        return self._eta_not

    @property
    def mu(self):
        """
        Angle between the lens axis and the direction to the caustic.

        Computed as ``arctan2(eta_not, (s - 1/s) / (1 + q))``. Includes a
        correction from the primary lens position to the center of mass.
        Used to calculate :attr:`alpha`.

        Returns
        -------
        float
            Angle in radians.
        """
        if self._mu is None:
            self._mu = np.arctan2(self.eta_not, (self.s - 1 / self.s) / (1 + self.q))
            # correction for primary --> COM

        return self._mu

    @property
    def phi(self):
        """
        Angle between the source trajectory and the line connecting the
        planetary caustic and the origin.

        Computed as ``arctan2(u_0, tau_pl)``. Used to calculate :attr:`alpha`.

        Returns
        -------
        float
            Angle in radians.
        """
        if self._phi is None:
            self._phi = np.arctan2(self.u_0, self.tau_pl)

        return self._phi

    @property
    def alpha(self):
        """
        Angle of the source trajectory relative to the binary axis, in degrees,
        for the upper caustic solution.

        Computed as ``180 - deg(phi - mu)``, where ``phi`` is the source
        trajectory angle and ``mu`` is the caustic offset angle.
        Normalized to (-360, 360] via :meth:`_correct_alpha`.

        Returns
        -------
        float
        """
        if self._alpha is None:
            alpha = 180. - np.rad2deg(self.phi - self.mu)
            self._alpha = self._correct_alpha(alpha)

        return self._alpha


class CloseLowerBinaryParameterEstimator(CloseUpperBinaryParameterEstimator):
    """
    Analytic parameter estimator for close binary lens models (lower caustic).

    Identical to :class:`CloseUpperBinaryParameterEstimator` except that
    ``alpha`` uses the lower caustic geometry, computed as
    ``180 - deg(phi + mu)``.
    """

    @property
    def alpha(self):
        """
        Angle of the source trajectory relative to the binary axis, in degrees,
        for the lower caustic solution.

        Computed as ``180 - deg(phi + mu)``, where ``phi`` is the source
        trajectory angle and ``mu`` is the caustic offset angle.
        Normalized to (-360, 360] via :meth:`_correct_alpha`.

        Returns
        -------
        float
        """
        if self._alpha is None:
            alpha = 180. - np.rad2deg(self.phi + self.mu)
            self._alpha = self._correct_alpha(alpha)

        return self._alpha


class CloseUpperBinaryGridSearchEstimator(WidePlanetGridSearchEstimator, CloseUpperBinaryParameterEstimator):
    """
    Grid search estimator for close binary lens models (upper caustic).

    Combines :class:`WidePlanetGridSearchEstimator` (chi2 grid search and
    Nelder-Mead refinement) with :class:`CloseUpperBinaryParameterEstimator`
    (close-topology analytic parameter estimates and upper caustic ``alpha``).
    """
    pass


class CloseLowerBinaryGridSearchEstimator(WidePlanetGridSearchEstimator, CloseLowerBinaryParameterEstimator):
    """
    Grid search estimator for close binary lens models (lower caustic).

    Combines :class:`WidePlanetGridSearchEstimator` (chi2 grid search and
    Nelder-Mead refinement) with :class:`CloseLowerBinaryParameterEstimator`
    (close-topology analytic parameter estimates and lower caustic ``alpha``).
    """
    pass


def get_close_params(params, q=None, rho=None):
    """
    Transform initial parameters into two close model parameters for a binary lens. One for upper and one for lower caustics.

    Arguments:
        params: *dictionary*
            Initial parameters.

            - 't_0' (*float*): Time of maximum magnification.
            - 'u_0' (*float*): Impact parameter.
            - 't_E' (*float*): Einstein crossing time.
            - 't_pl' (*float*): Time at which to compute the close model parameters.
            - 'dt' (*float*), optional: Duration of the anomaly
            - 'q' (*float*): trial value of q for calculating the caustic,
                default is 0.004
            - 'rho' (*float*): value of rho for the model. If 'dt' is specified,
                'rho' is calculated from 'dt'. If neither are specified,
                default is 0.001.

    Returns:
        lens1, lens2 : *tuple of BinaryLensParams*
            Two instances of BinaryLensParams representing close model parameters.
    """
    estimator_upper = CloseUpperBinaryParameterEstimator(params=params, q=q)
    estimator_lower = CloseLowerBinaryParameterEstimator(params=params, q=q)

    return estimator_upper.binary_params, estimator_lower.binary_params


class ClosePlanetParameterEstimator(WidePlanetParameterEstimator):
    """
    Analytic parameter estimator for close planet models.

    Extends :class:`WidePlanetParameterEstimator` to compute binary lens
    parameters appropriate for a close planet (s < 1), based on matching
    the dip in the light curve to the center of the demagnified region
    (mid-point between the planetary caustics). ``q`` is estimated from
    the anomaly duration and source trajectory geometry.

    Parameters
    ----------
    params : dict
        Anomaly light curve parameters as returned by
        :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
    limit : str, optional
        Method to use for estimating ``rho``. One of ``'GG97'`` (default),
        ``'dwarf'``, ``'giant'``, or ``'point'``.
    """

    @property
    def s(self):
        """
        Binary lens separation in Einstein radius units.

        Computed as ``|0.5 * (sqrt(u_pl^2 + 4) - u_pl)|``.
        This is the close-topology solution (s < 1).

        See Han 2006 https://ui.adsabs.harvard.edu/abs/2006ApJ...638.1080H/abstract

        Returns
        -------
        float
        """
        if self._s is None:
            u = self.u_pl
            self._s = np.abs(0.5 * (np.sqrt(u ** 2 + 4) - u))
        return self._s

    @property
    def q(self):
        """
        Planet-to-star mass ratio.

        Estimated from the anomaly duration and source trajectory geometry as
        ``(dt / t_E / 4)^2 * (s / u_0) * |sin(alpha)^3|``.

        See Hwang et al. 2022
        https://ui.adsabs.harvard.edu/abs/2022AJ....163...43H/abstract

        Returns
        -------
        float
        """
        if self._q is None:
            self._q = (self.params['dt'] / self.params['t_E'] / 4.) ** 2 * (self.s / self.u_0) * np.abs(
                (np.sin(np.deg2rad(self.alpha))) ** 3)
        return self._q

    @property
    def alpha(self):
        """
        Angle of the source trajectory relative to the binary axis, in degrees.

        Computed as ``-deg(arctan2(u_0, tau_pl))``.
        Normalized to (-360, 360] via :meth:`_correct_alpha`.

        Returns
        -------
        float
        """
        if self._alpha is None:
            alpha = np.arctan2(self.params['u_0'], self.tau_pl)
            alpha = np.rad2deg(-alpha)
            self._alpha = self._correct_alpha(alpha)

        return self._alpha


class ClosePlanetGridSearchEstimator(WidePlanetGridSearchEstimator, ClosePlanetParameterEstimator):
    """
    Grid search estimator for close planet models.

    Combines :class:`WidePlanetGridSearchEstimator` (chi2 grid search and
    Nelder-Mead refinement) with :class:`ClosePlanetParameterEstimator`
    (close-topology analytic parameter estimates).
    """

    @property
    def s_values(self):
        """
        Grid values of s for the grid search.

        If ``s_grid`` was provided at construction, returns that directly.
        Otherwise, constructs a uniform grid of ``n_s=7`` points centered
        on the analytic ``s`` estimate with step size ``d_s=0.05 * s``,
        then filters to only include values less than 1. The wider default
        grid (compared to :class:`WidePlanetGridSearchEstimator`) accounts
        for the broader range of possible s values in non-caustic-crossing
        close planet geometries.

        Returns
        -------
        numpy.ndarray
        """
        if self._s_grid is not None:
            return self._s_grid

        d_s = self.d_s if self.d_s is not None else 0.05 * self.s
        n_s = self.n_s if self.n_s is not None else 7
        s_offset = np.arange(n_s) - (n_s - 1) / 2
        s_values = self.s + s_offset * d_s
        return s_values[s_values < 1.]


def model_pspl_mag_at_pl(params):
    """
    Compute the PSPL magnification at the anomaly time.

    Parameters
    ----------
    params : dict
        Anomaly light curve parameters as returned by
        :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
        Required keys:

        - ``'t_0'`` : float, time of maximum magnification.
        - ``'u_0'`` : float, impact parameter.
        - ``'t_E'`` : float, Einstein crossing time.
        - ``'t_pl'`` : float, time of the anomaly.

    Returns
    -------
    float
        PSPL magnification at ``t_pl``.
    """
    model1 = MulensModel.Model({'t_0': params['t_0'],
                                'u_0': params['u_0'],
                                't_E': params['t_E']})
    return model1.get_magnification(params['t_pl'])


class BinarySourceParams():
    """
    Container for binary source model parameters.

    Parameters
    ----------
    ulens : dict
        PSPL parameter dictionary with keys ``'t_0_1'``, ``'u_0_1'``,
        ``'t_0_2'``, ``'u_0_2'``, ``'t_E'``.

    Attributes
    ----------
    ulens : dict
        Binary source parameter dictionary.
    source_flux_ratio : float or None
        Source flux ratio. Set by :meth:`set_source_flux_ratio`.
    """

    def __init__(self, ulens):
        self.ulens = ulens
        self.source_flux_ratio = None

    def set_source_flux_ratio(self, params):
        """
        Set the source flux ratio from anomaly light curve parameters.

        See Gaudi 1998 equation 2.5
        https://ui.adsabs.harvard.edu/abs/1998ApJ...506..533G/abstract

        Parameters
        ----------
        params : dict
            Anomaly light curve parameters as returned by
            :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
            Required keys:

            - ``'t_0'`` : float, time of maximum magnification.
            - ``'u_0'`` : float, impact parameter.
            - ``'t_E'`` : float, Einstein crossing time.
            - ``'t_pl'`` : float, time of the anomaly.
            - ``'dt'`` : float, duration of the anomaly.
            - ``'dmag'`` : float, magnitude difference of the perturbation.

        Returns
        -------
        None
        """
        A1 = model_pspl_mag_at_pl(params)
        u_0_2 = params["dt"] / (12 ** 0.5 * params["t_E"])
        e = params["dmag"] * u_0_2 * A1
        self.source_flux_ratio = e


def get_binary_source_params(params):
    """
    Transform initial anomaly parameters into binary source model parameters.

    Parameters
    ----------
    params : dict
        Anomaly light curve parameters as returned by
        :meth:`AnomalyPropertyEstimator.get_anomaly_lc_parameters`.
        Required keys:

        - ``'t_0'`` : float, time of maximum magnification.
        - ``'u_0'`` : float, impact parameter.
        - ``'t_E'`` : float, Einstein crossing time.
        - ``'t_pl'`` : float, time of the anomaly.
        - ``'dt'`` : float, duration of the anomaly.
        - ``'dmag'`` : float, magnitude difference of the perturbation.

    Returns
    -------
    :class:`BinarySourceParams`
        Binary source model parameters with ``source_flux_ratio`` set.
    """
    u_0_2 = params["dt"] / (12 ** 0.5 * params["t_E"])
    new_params = {'t_0_1': params['t_0'],
                  'u_0_1': params['u_0'],
                  't_0_2': params['t_pl'],
                  'u_0_2': u_0_2,
                  't_E': params['t_E']}
    out = BinarySourceParams(new_params)
    out.set_source_flux_ratio(params)
    return out


class AnomalyPropertyEstimator():
    """
    Estimates photometric anomaly properties from PSPL residuals.

    Identifies the time, duration, and magnitude of a photometric anomaly
    by computing residuals relative to a PSPL model and finding the dominant
    extremum within the anomaly window defined by the AnomalyFinderGridSearch
    results.

    Parameters
    ----------
    datasets : MulensModel.MulensData or list of MulensModel.MulensData
        Photometric datasets.
    pspl_params : dict
        PSPL model parameters. Required keys: ``'t_0'``, ``'u_0'``, ``'t_E'``.
    af_results : dict
        Best point from the AnomalyFinderGridSearch. Required keys:
        ``'t_0'``, ``'t_eff'``.
    model_config : :class:`.mulens_object_config.ModelConfig`, optional
        Configuration for Model construction. If None, a default
        ``ModelConfig`` is used (no coords, no limb darkening).
    event_config : :class:`.mulens_object_config.EventConfig`, optional
        Configuration for Event construction. If None, a default
        ``EventConfig`` is used (no coords, no flux fixing).
    n_mask : int or float, optional
        Half-width of the anomaly window in units of ``t_eff``. Default is 3.
    importance_threshold : float, optional
        If the amplitude of a negative peak is less than this fraction of
        the dominant positive peak, it is considered unimportant.
        Default is 0.2.
    """
    # TODO: The old version revised the PSPL parameters after masking the anomaly.
    # Could consider whether it would be a good idea to reimplement that.

    def __init__(self,
                 datasets=None, pspl_params=None, af_results=None,
                 model_config=None, event_config=None,
                 n_mask=3,
                 importance_threshold=0.2):

        if isinstance(datasets, MulensModel.MulensData):
            datasets = [datasets]

        self.datasets = datasets
        self.pspl_params = pspl_params
        self.af_results = af_results
        self.model_config = model_config if model_config is not None else ModelConfig()
        self.event_config = event_config if event_config is not None else EventConfig()
        self.n_mask = n_mask
        self.importance_threshold = importance_threshold

        self.anom_t_range_af = self.af_results['t_0'] + self.n_mask * np.array(
            [-1, 1]) * self.af_results['t_eff']

        self._peak_index = None
        self._peak_dflux = None
        self._t_start = None
        self._t_stop = None
        self.all_peaks = None

        self._pspl_event = None
        self._source_flux = None
        self._blend_flux = None

        self._anom_index = None
        self._sorted_index = None
        self._times = None
        self._scaled_fluxes = None
        self._scaled_residuals = None
        self._chi2s = None
        self._expected_model_fluxes = None

    def get_pspl_event(self):
        """
        Create a MulensModel.Event for the PSPL model.

        Coordinates and flux-fixing are applied via ``event_config``.

        Returns
        -------
        MulensModel.Event
            Event with fluxes fitted.
        """
        model = self.model_config.build(parameters=self.pspl_params)
        event = self.event_config.build(
            model=model,
            datasets=self.datasets,
        )
        event.fit_fluxes()
        return event

    def set_anom_prop(self):
        """
        Compute and cache anomaly properties if not already set.

        Calls :meth:`find_extremum` with ``method='rolling'`` and stores the
        results in ``_peak_dflux``, ``_peak_index``, ``_t_start``, and
        ``_t_stop``.

        Returns
        -------
        None
        """
        if self._peak_dflux is None:
            self._peak_dflux, self._peak_index, self._t_start, self._t_stop = self.find_extremum(
                method='rolling')

    def get_anom_prop(self):
        """
        Return anomaly properties, computing them first if necessary.

        Calls :meth:`set_anom_prop` if ``peak_dflux``, ``t_start``, or
        ``t_stop`` are not yet set.

        Returns
        -------
        peak_dflux : float
            Peak flux deviation of the anomaly.
        peak_index : int
            Index into :attr:`sorted_times` of the dominant peak.
        t_start : float
            Start time of the anomaly.
        t_stop : float
            End time of the anomaly.
        peak_width : float
            Duration of the anomaly (``t_stop - t_start``).
        """
        if (self.peak_dflux is None) or (self.t_start is None) or (self.t_stop is None):
            self.set_anom_prop()

        return self.peak_dflux, self.peak_index, self.t_start, self.t_stop, self.peak_width

    def _find_extremum_with_simple_line(self):
        """
        Estimate the anomaly extremum using a simple linear interpolation.

        Fallback method used when the anomaly window contains too few points
        for rolling-mean smoothing. Identifies the peak as the point of maximum
        chi2, then estimates ``t_start`` and ``t_stop`` by linear interpolation
        between the peak and the first/last points in the window.

        Returns
        -------
        peak_dflux : float
            Peak flux deviation of the anomaly.
        peak_index : int
            Index into :attr:`sorted_times` of the dominant peak.
        t_start : float
            Estimated start time of the anomaly.
        t_stop : float
            Estimated end time of the anomaly.
        """
        peak_index = np.nanargmax(self.chi2s)
        peak_dflux = self.residuals[peak_index]
        t_start, t_stop = None, None
        for i in [1, -1]:
            slope = (self.sorted_times[peak_index] - self.sorted_times[i]) / (self.peak_dflux - self.residuals[i])
            intercept = self.sorted_times[peak_index] - slope * peak_dflux
            t = slope * peak_dflux / 2. + intercept
            if i == 1:
                t_start = t
            else:
                t_stop = t

        return peak_dflux, peak_index, t_start, t_stop

    def _get_window_size(self):
        """
        Compute the rolling mean window size based on the number of anomaly points.

        Scales the window to roughly 10% of the anomaly points for small windows,
        decreasing to ~1% for large windows.

        Returns
        -------
        int
            Window size for use with :meth:`_find_extremum_with_rolling_mean`.
        """
        n_pts = np.sum(self.anom_index)

        if n_pts < 10:
            window_size = 1
        elif n_pts < 50:
            window_size = int(np.floor(n_pts / 10))
        elif n_pts < 100:
            window_size = int(np.floor(n_pts / 20))
        elif n_pts < 500:
            window_size = int(np.floor(n_pts / 50))
        else:
            window_size = int(np.floor(n_pts / 100))

        return window_size

    def _find_all_extrema(self, res_rolling_mean, prominence):
        """
        Find all local extrema in the smoothed residuals.

        Uses ``scipy.signal.find_peaks`` on both the positive and negative
        signal. If no peaks are found, falls back to the global extremum.

        Parameters
        ----------
        res_rolling_mean : numpy.ndarray
            Smoothed residuals from the rolling mean.
        prominence : float
            Minimum prominence required for a peak to be detected.

        Returns
        -------
        list of dict
            Each dict has keys ``'peak_index'`` (int) and ``'peak_dflux'``
            (float).
        """
        pos_indices, _ = find_peaks(res_rolling_mean, prominence=prominence)
        neg_indices, _ = find_peaks(-res_rolling_mean, prominence=prominence)

        peaks = []
        for idx in pos_indices:
            peaks.append({'peak_index': idx, 'peak_dflux': res_rolling_mean[idx]})
        for idx in neg_indices:
            peaks.append({'peak_index': idx, 'peak_dflux': res_rolling_mean[idx]})

        if len(peaks) == 0:
            max_idx = np.argmax(res_rolling_mean)
            min_idx = np.argmin(res_rolling_mean)
            if abs(res_rolling_mean[max_idx]) >= abs(res_rolling_mean[min_idx]):
                peaks.append({'peak_index': max_idx, 'peak_dflux': res_rolling_mean[max_idx]})
            else:
                peaks.append({'peak_index': min_idx, 'peak_dflux': res_rolling_mean[min_idx]})

        return peaks

    def _select_dominant_peak(self, peaks):
        """
        Select the dominant peak from a list of candidates.

        Uses :attr:`importance_threshold` to handle mixed-sign cases.

        Parameters
        ----------
        peaks : list of dict
            Candidates as returned by :meth:`_find_all_extrema`. Each dict
            has keys ``'peak_index'`` and ``'peak_dflux'``.

        Returns
        -------
        dict
            The dominant peak dict with keys ``'peak_index'`` and
            ``'peak_dflux'``.

        Notes
        -----
        Selection logic:

        1. All positive peaks: return the largest positive peak.
        2. All negative peaks: return the largest negative peak.
        3. Mixed sign:

           a. If ``|biggest_neg| > importance_threshold * |biggest_pos|``,
              both signs are considered important; return the peak with the
              largest absolute amplitude.
           b. Otherwise, the negative peak is considered unimportant;
              return the largest positive peak.
        """
        pos_peaks = [p for p in peaks if p['peak_dflux'] > 0]
        neg_peaks = [p for p in peaks if p['peak_dflux'] < 0]

        if len(pos_peaks) == 0:
            return max(neg_peaks, key=lambda p: abs(p['peak_dflux']))
        elif len(neg_peaks) == 0:
            return max(pos_peaks, key=lambda p: abs(p['peak_dflux']))
        else:
            biggest_pos = max(pos_peaks, key=lambda p: abs(p['peak_dflux']))
            biggest_neg = max(neg_peaks, key=lambda p: abs(p['peak_dflux']))
            if abs(biggest_neg['peak_dflux']) <= self.importance_threshold * abs(biggest_pos['peak_dflux']):
                return biggest_pos
            else:
                return biggest_neg

    def _find_extremum_with_rolling_mean(self):
        """
        Estimate the anomaly extremum using a rolling mean.

        Smooths the residuals with a boxcar kernel of size from
        :meth:`_get_window_size`, estimates noise via median absolute deviation,
        finds all peaks with prominence ≥ 3σ via :meth:`_find_all_extrema`,
        selects the dominant peak via :meth:`_select_dominant_peak`, and
        estimates ``t_start``/``t_stop`` as the times where the smoothed
        residuals cross the half-maximum of the dominant peak.

        Falls back to :meth:`_find_extremum_with_simple_line` if the window
        size equals or exceeds the number of anomaly points.

        Returns
        -------
        peak_dflux : float
            Peak flux deviation of the anomaly.
        peak_index : int
            Index into :attr:`sorted_times` of the dominant peak.
        t_start : float
            Start time of the anomaly (half-maximum crossing).
        t_stop : float
            End time of the anomaly (half-maximum crossing).
        """
        window_size = self._get_window_size()
        kernel = np.ones(window_size) / window_size

        if (window_size > 0) and (window_size < np.sum(self.anom_index)):
            res_rolling_mean = np.convolve(self.residuals, kernel, mode='same')

            noise = scipy.stats.median_abs_deviation(self.residuals)
            prominence = 3. * noise

            self.all_peaks = self._find_all_extrema(res_rolling_mean, prominence=prominence)
            dominant_peak = self._select_dominant_peak(self.all_peaks)

            peak_index = dominant_peak['peak_index']
            peak_dflux = dominant_peak['peak_dflux']

            if peak_dflux > 0:
                half_anomaly = res_rolling_mean > (peak_dflux / 2.)
            else:
                half_anomaly = res_rolling_mean < (peak_dflux / 2.)

            t_start = np.min(self.sorted_times[half_anomaly])
            t_stop = np.max(self.sorted_times[half_anomaly])

            return peak_dflux, peak_index, t_start, t_stop
        else:
            return self._find_extremum_with_simple_line()

    def find_extremum(self, method=None):
        """
        Find the dominant extremum in the anomaly window.

        Dispatches to the appropriate algorithm based on ``method``.

        Parameters
        ----------
        method : str, optional
            Algorithm to use. Currently only ``'rolling'`` is supported,
            which calls :meth:`_find_extremum_with_rolling_mean`.

        Returns
        -------
        peak_dflux : float
            Peak flux deviation of the anomaly.
        peak_index : int
            Index into :attr:`sorted_times` of the dominant peak.
        t_start : float
            Start time of the anomaly.
        t_stop : float
            End time of the anomaly.
        """
        if method == 'rolling':
            return self._find_extremum_with_rolling_mean()

    def get_anomaly_lc_parameters(self):
        """
        Return the full set of anomaly light curve parameters.

        Combines the PSPL parameters with the estimated anomaly properties
        (``dmag``, ``dt``, ``t_pl``). The returned dictionary is the standard
        ``params`` input expected by all parameter estimator classes in this
        module.

        Returns
        -------
        dict
            PSPL parameters plus:

            - ``'dmag'`` : float, magnitude difference of the anomaly.
            - ``'dt'`` : float, duration of the anomaly (``t_stop - t_start``).
            - ``'t_pl'`` : float, midpoint time of the anomaly.
        """
        self.set_anom_prop()
        params = {key: value for key, value in self.pspl_params.items()}
        params['dmag'] = self.dmag
        params['dt'] = self.t_stop - self.t_start
        params['t_pl'] = np.mean((self.t_start, self.t_stop))

        return params

    def _plot_peak_lines(self):
        """Draw vertical lines at peak_time, t_start, and t_stop on the current axes."""
        plt.axvline(self.peak_time, color='darkgray', zorder=10, linestyle=':')
        plt.axvline(self.t_start, color='darkgray')
        plt.axvline(self.t_stop, color='darkgray')

    def _plot_peak_lines_res(self):
        """Draw peak lines and a horizontal line at peak_dflux; overlay all_peaks as scatter points."""
        self._plot_peak_lines()
        plt.axhline(self.peak_dflux, color='darkgray', linestyle=':')
        if self.all_peaks is not None:
            for peak in self.all_peaks:
                plt.scatter(
                    self.sorted_times[peak['peak_index']], peak['peak_dflux'],
                    marker='d', color='red', zorder=5)

    def _plot_af_lines(self):
        """Draw vertical lines at the AnomalyFinder t_eff boundaries on the current axes."""
        plt.axvline(self.af_results['t_0'] +
                    self.af_results['t_eff'], color='black')
        plt.axvline(self.af_results['t_0'] -
                    self.af_results['t_eff'], color='black')

    def _setup_anom_xaxis(self):
        """Set x-axis limits to ±5 t_eff around the AnomalyFinder t_0 and label as 'time'."""
        plt.xlim(self.af_results['t_0'] + 5. * np.array([-1, 1]) *
                 self.af_results['t_eff'])
        plt.xlabel('time')

    def plot_residuals(self):
        """
        Plot PSPL residuals in the anomaly window.

        Scatter plot of residuals vs. time with peak and AnomalyFinder boundary
        lines overlaid via :meth:`_plot_peak_lines_res` and
        :meth:`_plot_af_lines`.
        """
        plt.figure()
        plt.axhline(0, color='black')
        plt.scatter(self.sorted_times, self.residuals)
        self._plot_peak_lines_res()
        self._plot_af_lines()
        self._setup_anom_xaxis()
        plt.ylabel('res')

    def plot_anomaly(self):
        """
        Plot the PSPL model and data in the anomaly window.

        Shows the data, PSPL model, and the estimated peak anomaly magnitude,
        with peak and AnomalyFinder boundary lines overlaid via
        :meth:`_plot_peak_lines` and :meth:`_plot_af_lines`.
        """
        plt.figure()
        self.pspl_event.plot_data()
        self.pspl_event.plot_model(color='black', zorder=5)
        peak_anom_mag = MulensModel.Utils.get_mag_from_flux(
            self.expected_model_fluxes[self.peak_index] + self.peak_dflux)
        plt.scatter(self.peak_time, peak_anom_mag, marker='d', color='darkgray', zorder=10)

        self._plot_peak_lines()
        self._plot_af_lines()
        self._setup_anom_xaxis()

        plt.ylabel('mag')

    @property
    def peak_dflux(self):
        """Peak flux deviation of the dominant anomaly, set by :meth:`set_anom_prop`."""
        return self._peak_dflux

    @property
    def peak_index(self):
        """Index into :attr:`sorted_times` of the dominant peak, set by :meth:`set_anom_prop`."""
        return self._peak_index

    @property
    def peak_time(self):
        """Time of the dominant peak (``sorted_times[peak_index]``)."""
        return self.sorted_times[self.peak_index]

    @property
    def t_start(self):
        """Start time of the anomaly, set by :meth:`set_anom_prop`."""
        return self._t_start

    @property
    def t_stop(self):
        """End time of the anomaly, set by :meth:`set_anom_prop`."""
        return self._t_stop

    @property
    def dmag(self):
        """
        Magnitude difference between the anomaly peak and the PSPL model.

        Computed as ``mag(model_flux + peak_dflux) - mag(model_flux)`` at the
        peak index, where ``model_flux`` is the expected PSPL flux.

        Returns
        -------
        float
        """
        expected_mag = MulensModel.Utils.get_mag_from_flux(
            self.expected_model_fluxes[self.peak_index])
        peak_anom_mag = MulensModel.Utils.get_mag_from_flux(
            self.expected_model_fluxes[self.peak_index] + self.peak_dflux)

        return peak_anom_mag - expected_mag

    @property
    def peak_width(self):
        """Duration of the anomaly (``t_stop - t_start``)."""
        return self.t_stop - self.t_start

    @property
    def anom_index(self):
        """
        Boolean mask selecting data points within the anomaly window.

        The window is ``t_0 ± n_mask * t_eff`` from :attr:`af_results`,
        computed at construction as ``anom_t_range_af``.

        Returns
        -------
        numpy.ndarray of bool
        """
        if self._anom_index is None:
            self._anom_index = (self.times > self.anom_t_range_af[0]) & (self.times < self.anom_t_range_af[1])

        return self._anom_index

    @property
    def sorted_index(self):
        """
        Indices that sort :attr:`times` within the anomaly window.

        Returns
        -------
        numpy.ndarray of int
        """
        if self._sorted_index is None:
            self._sorted_index = np.argsort(self.times[self.anom_index])

        return self._sorted_index

    @property
    def times(self):
        """
        Observation times from all datasets, concatenated.

        Returns
        -------
        numpy.ndarray
        """
        if self._times is None:
            self._times = np.hstack([dataset.time for dataset in self.pspl_event.datasets])

        return self._times

    @property
    def sorted_times(self):
        """
        Observation times within the anomaly window, sorted in time order.

        Returns
        -------
        numpy.ndarray
        """
        return self.times[self.anom_index][self.sorted_index]

    @property
    def pspl_event(self):
        """PSPL MulensModel.Event with fluxes fitted, computed lazily via :meth:`get_pspl_event`."""
        if self._pspl_event is None:
            self._pspl_event = self.get_pspl_event()

        return self._pspl_event

    @property
    def source_flux(self):
        """Reference source flux from the PSPL fit, from :attr:`pspl_event`."""
        if self._source_flux is None:
            self._source_flux, foo = self.pspl_event.get_ref_fluxes()

        return self._source_flux

    @property
    def blend_flux(self):
        """Reference blend flux from the PSPL fit, from :attr:`pspl_event`."""
        if self._blend_flux is None:
            foo, self._blend_flux = self.pspl_event.get_ref_fluxes()

        return self._blend_flux

    @property
    def scaled_fluxes(self):
        """
        Scaled fluxes within the anomaly window, sorted in time order.

        Concatenated from all datasets via ``pspl_event.get_scaled_fluxes()``,
        then filtered by :attr:`anom_index` and ordered by :attr:`sorted_index`.

        Returns
        -------
        numpy.ndarray
        """
        if self._scaled_fluxes is None:
            self._scaled_fluxes = np.hstack(
                [np.array(flux) for (flux, err) in self.pspl_event.get_scaled_fluxes()])[self.anom_index][
                self.sorted_index]

        return self._scaled_fluxes

    @property
    def residuals(self):
        """
        Flux residuals within the anomaly window (``scaled_fluxes - expected_model_fluxes``).

        Returns
        -------
        numpy.ndarray
        """
        if self._scaled_residuals is None:
            self._scaled_residuals = self.scaled_fluxes - self.expected_model_fluxes

        return self._scaled_residuals

    @property
    def chi2s(self):
        """
        Chi2 per point within the anomaly window, sorted in time order.

        Returns
        -------
        numpy.ndarray
        """
        if self._chi2s is None:
            self._chi2s = np.hstack(self.pspl_event.get_chi2_per_point())[self.anom_index][self.sorted_index]

        return self._chi2s

    @property
    def expected_model_fluxes(self):
        """
        Expected PSPL model fluxes at :attr:`sorted_times`.

        Computed as ``source_flux * magnification + blend_flux``.

        Returns
        -------
        numpy.ndarray
        """
        if self._expected_model_fluxes is None:
            self._expected_model_fluxes = self.source_flux * self.pspl_event.model.get_magnification(
                self.sorted_times) + self.blend_flux

        return self._expected_model_fluxes
