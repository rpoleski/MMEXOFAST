import MulensModel
import numpy as np
import sfit_minimizer as sfit
import emcee
from multiprocessing import Pool, cpu_count
import os

from .estimate_params import WidePlanetEnsembleInitializer
from .mulens_object_config import ModelConfig, EventConfig


class MinimalResults:
    def __init__(
        self,
        emcee_percentiles=None,
        x=None,
        sigmas=None,
        success=None,
        msg=None,
        parameters_to_fit=None,
    ):
        """
        Parameters
        ----------
        emcee_percentiles : array-like of shape (n_parameters, 3), optional
            The 16th, 50th, and 84th percentiles for each parameter in
            parameters_to_fit. If provided, x and sigmas cannot also be given.
        x : array-like, optional
            Best-fit parameter values. Derived from emcee_percentiles
            (50th percentile column) if emcee_percentiles is provided.
        sigmas : array-like, optional
            Parameter uncertainties. Derived from emcee_percentiles as the
            mean of (p50 - p16) and (p84 - p50) per parameter if
            emcee_percentiles is provided.
        success : bool, optional
            Whether the fit was successful.
        msg : str, optional
            A message describing the fit result.
        parameters_to_fit : list, optional
            Names of the parameters that were fitted.
        """
        if emcee_percentiles is not None:
            if x is not None:
                raise ValueError(
                    "Cannot provide both 'emcee_percentiles' and 'x'. "
                    "'x' is derived from 'emcee_percentiles'."
                )
            if sigmas is not None:
                raise ValueError(
                    "Cannot provide both 'emcee_percentiles' and 'sigmas'. "
                    "'sigmas' is derived from 'emcee_percentiles'."
                )

        self.emcee_percentiles = (
            np.array(emcee_percentiles) if emcee_percentiles is not None else None
        )
        self.success = success
        self.msg = msg
        self.parameters_to_fit = parameters_to_fit

        if self.emcee_percentiles is not None:
            p16 = self.emcee_percentiles[0, :]
            p50 = self.emcee_percentiles[1, :]
            p84 = self.emcee_percentiles[2, :]
            self.x = p50
            self.sigmas = ((p50 - p16) + (p84 - p50)) / 2
        else:
            self.x = x
            self.sigmas = sigmas

    def __repr__(self):
        return (
            f"MinimalResults(\n"
            f"  parameters_to_fit={self.parameters_to_fit},\n"
            f"  x={self.x},\n"
            f"  sigmas={self.sigmas},\n"
            f"  success={self.success},\n"
            f"  msg={self.msg}\n"
            f")"
        )


class MulensFitter():
    """
    Parent class for microlensing model fitters.

    Parameters
    ----------
    datasets : list
        List of MulensModel.MulensData objects.
    initial_model_params : dict
        Initial parameters of the model.
    parameters_to_fit : list, optional
        Parameters to be fitted. If None, all keys in initial_model_params are fitted.
    sigmas : dict, optional
        Dict mapping parameter names to step sizes. Parameters
        not in the dict default to ``None``.
    mag_methods : list, optional
        Magnification methods specification; see
        MulensModel.Model.set_magnification_methods. Passed directly to
        ``get_model()`` since it varies per model type.
    model_config : ModelConfig, optional
        Configuration for Model construction (coords, limb darkening, etc.).
        If None, a default ``ModelConfig`` is used.
    event_config : EventConfig, optional
        Configuration for Event construction (coords, flux fixing, etc.).
        If None, a default ``EventConfig`` is used.
    verbose : bool, optional
        If True, print progress information. Default is False.
    pool : multiprocessing.Pool, optional
        Pool for parallel computation.
    """

    def __init__(
            self, datasets=None, initial_model_params=None, parameters_to_fit=None, sigmas=None,
            mag_methods=None, model_config=None, event_config=None,
            verbose=False, pool=None):
        self._initial_model = None
        self._best = None
        self._results = None

        self.datasets = datasets

        self.initial_model_params = initial_model_params
        self.parameters_to_fit = parameters_to_fit
        self.sigmas = sigmas

        self.mag_methods = mag_methods
        self.model_config = (
            model_config if model_config is not None else ModelConfig()
        )
        self.event_config = (
            event_config if event_config is not None else EventConfig()
        )

        self.verbose = verbose
        self.pool = pool

    def run(self):
        """
        Run the fitter. Implemented by subclasses.

        Notes
        -----
        This method is not formally declared as abstract but is intended to be
        overridden by subclasses. Consider using ``abc.abstractmethod`` to enforce
        this, consistent with ``set_event_parameters()`` and
        ``make_starting_vector()`` in ``AnomalyFitter``.
        """
        pass

    def get_model(self):
        """
        Create a MulensModel.Model with best-fit parameters, or initial
        parameters if no fit has been run yet.

        Magnification methods and limb darkening coefficients are applied
        via ``model_config``, which is built from the constructor arguments
        at initialization time.

        Returns
        -------
        MulensModel.Model
            Configured model with best-fit or initial parameters.

        See Also
        --------
        get_event : Creates a MulensModel.Event using this model.
        """
        if self.best is not None:
            params = dict(self.best)
            params.pop('chi2', None)
        else:
            params = self.initial_model_params

        return self.model_config.build(
            parameters=params,
            magnification_methods=self.mag_methods,
        )

    def get_event(self):
        """
        Create a MulensModel.Event using the current datasets and model.

        Coordinates and flux-fixing are applied via ``event_config``, which
        is built from the constructor arguments at initialization time.

        Returns
        -------
        MulensModel.Event
            Event constructed from the current fitter state.

        See Also
        --------
        get_model : Creates the MulensModel.Model used by this event.
        """
        return self.event_config.build(
            model=self.get_model(),
            datasets=self.datasets,
        )

    def get_diagnostic_str(self):
        """
        Build a diagnostic string summarising the current event fit.

        Calls ``fit_fluxes()`` on the event before building the string. The
        returned string includes the model parameters, and for each dataset: the
        label, number of good data points, chi2, source flux(es), and blend flux.

        Returns
        -------
        str
            Formatted string containing event and dataset fit information.

        Notes
        -----
        Despite the name, this method does not print anything. Use
        ``print(get_diagnostic_str())`` to print the output.
        """
        event = self.get_event()
        event.fit_fluxes()
        msg = f'\n---- Event Info ----\nModel:\n{event.model}\n\nDatasets:'
        msg += '\n{0:20} {1:>4} {2:>12} {3} {4}'.format('Label', 'N_good', 'chi2', 'f_source', 'f_blend')
        for i, dataset in enumerate(event.datasets):
            msg += ('\n{0:20} {1:4} {2:12.2f} {3} {4}'.format(
                dataset.plot_properties['label'], np.sum(dataset.good),
                event.get_chi2_for_dataset(i),
                event.fits[i].source_fluxes,
                event.fits[i].blend_flux))

        msg += '\n--------------------\n'
        return msg

    @property
    def best(self):
        """
        Best-fit model parameters and chi2.

        Returns
        -------
        dict or None
            Dictionary of best-fit model parameter names and values, with an
            additional ``'chi2'`` key. Returns None if no fit has been run yet.
        """
        return self._best

    @best.setter
    def best(self, params_dict):
        self._best = params_dict

    @property
    def results(self):
        """
        Full results object from the fitter.

        Returns
        -------
        dict or None
            Full results from the fitting routine. The structure depends on the
            subclass: for example, ``SFitFitter`` stores the result object returned
            by ``sfit.minimize()``. Returns None if no fit has been run yet.

        Notes
        -----
        For the best-fit model parameters specifically, use ``best`` instead.
        """
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    @property
    def initial_model_params(self):
        """
        Initial model parameters used as the starting point for the fit.

        Parameters
        ----------
        params_dict : dict or None
            Dictionary of model parameter names and values. Must be a dict or
            None; raises ValueError otherwise.

        Returns
        -------
        dict or None
            Dictionary of model parameter names and values.

        Raises
        ------
        ValueError
            If set with a value that is neither None nor a dict.
        """
        return self._initial_model_params

    @initial_model_params.setter
    def initial_model_params(self, params_dict):
        if (params_dict is not None) and (not isinstance(params_dict, dict)):
            raise ValueError('initial_model must be set with either *None* or *dict*.')

        self._initial_model_params = params_dict

    @property
    def parameters_to_fit(self):
        """
        List of model parameters to be fitted.

        If not explicitly set, defaults to all keys in ``initial_model_params``.

        Parameters
        ----------
        params_dict : list, tuple, or None
            Names of parameters to fit. Must be a list, tuple, or None; raises
            ValueError otherwise. If None, all keys in ``initial_model_params``
            will be fitted.

        Returns
        -------
        list
            Names of parameters to be fitted.

        Raises
        ------
        ValueError
            If set with a value that is neither None, a list, nor a tuple.
        """
        if self._parameters_to_fit is None:
            self._parameters_to_fit = list(self.initial_model_params.keys())

        return self._parameters_to_fit

    @parameters_to_fit.setter
    def parameters_to_fit(self, params_dict):
        if (params_dict is not None) and (not isinstance(params_dict, (list, tuple))):
            raise ValueError('parameters_to_fit must be set with either *None* or *list* or *tuple*.')

        self._parameters_to_fit = params_dict


class SFitFitter(MulensFitter):
    """
    Fit a point lens model to the data using the SFit method.

    Wraps ``sfit.minimize()`` with a ``PointLensSFitFunction``. First attempts
    the fit with an adaptive step size; if unsuccessful, retries with a fixed
    step size of 0.001 and a maximum of 10000 iterations.

    All parameters are inherited from ``MulensFitter``.

    See Also
    --------
    MulensFitter : Parent class defining all constructor parameters.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        """
        Fit the point lens model using the SFit method.

        Constructs an initial guess vector from ``initial_model_params`` and the
        source and blend fluxes from an initial ``fit_fluxes()`` call. First
        attempts minimization with an adaptive step size; if unsuccessful, retries
        with a fixed step size of 0.001 and a maximum of 10000 iterations.

        On completion, sets ``results`` to the full ``sfit.minimize()`` result
        object and ``best`` to the best-fit model parameters plus ``'chi2'``.

        Notes
        -----
        ``event.fits[i].source_flux`` should be ``event.fits[i].source_fluxes``
        for MMv3, where source fluxes are returned as an array. The way the value
        is subsequently appended to ``initial_guess`` may also need revisiting.
        See also ``get_diagnostic_str()``, which correctly uses ``source_fluxes``.
        """
        event = self.get_event()
        event.fit_fluxes()

        my_func = sfit.mm_funcs.PointLensSFitFunction(
            event, self.parameters_to_fit)

        initial_guess = [self.initial_model_params[key] for key in self.parameters_to_fit]
        for i in range(len(self.datasets)):
            initial_guess.append(event.fits[i].source_flux)
            initial_guess.append(event.fits[i].blend_flux)

        result = sfit.minimize(
            my_func, x0=initial_guess, tol=1e-5,
            options={'step': 'adaptive'}, verbose=self.verbose)

        if self.verbose:
            print(result)

        if not result.success:
            result = sfit.minimize(
                my_func, x0=initial_guess, tol=1e-5, max_iter=10000,
                options={'step': 0.001}, verbose=self.verbose)
            if self.verbose:
                print(result)


        self.results = result
        best = my_func.event.model.parameters.parameters
        best['chi2'] = my_func.event.get_chi2()
        self.best = best


class EmceeLCFitter(MulensFitter):
    """
    Pure emcee mechanism for fitting microlensing light curves.

    Extends :class:`MulensFitter` with MCMC sampling via emcee.  The starting
    ensemble is built by Gaussian perturbation of ``initial_guess`` scaled by
    ``sigmas``.  All domain-specific policy — parameter estimation, ensemble
    initialisation, log-space defaults — lives in subclasses.

    Parameters
    ----------
    initial_guess : dict, optional
        Starting parameter values keyed by parameter name.  Keys may carry a
        ``log_`` prefix (e.g. ``'log_rho'``) to indicate sampling in log10
        space; :meth:`initialize_event` converts them back to linear space.
        If ``parameters_to_fit`` is not supplied it defaults to
        ``list(initial_guess.keys())``.
    emcee_settings : dict, optional
        Settings for the emcee sampler.  Missing keys are filled from
        :attr:`default_emcee_settings`.  Valid keys: ``'n_walkers'``,
        ``'n_burn'``, ``'n_steps'``, ``'acceptance_fraction'``, and
        optionally ``'temperature'``.  ``'n_dim'`` is added automatically
        from ``len(parameters_to_fit)`` once that list is finalised.

    Class Attributes
    ----------------
    default_emcee_settings : dict
        ``{'n_walkers': 40, 'n_burn': 500, 'n_steps': 1000,
        'acceptance_fraction': 0.1}``

    Notes
    -----
    After ``event.get_chi2()`` has been evaluated the ``MulensModel.Event``
    object cannot be pickled.  :meth:`__getstate__` resets ``_event`` to
    ``None`` before serialisation; call :meth:`initialize_event` again after
    unpickling if needed.

    Subclasses that finalise ``parameters_to_fit`` *after*
    ``super().__init__()`` are responsible for updating ``'n_dim'`` in
    :attr:`emcee_settings` themselves.

    All other parameters (``datasets``, ``sigmas``, ``model_config``,
    ``event_config``, ``pool``, …) are inherited from :class:`MulensFitter`.
    """

    default_emcee_settings = {
        'n_walkers': 40, 'n_burn': 500, 'n_steps': 1000,
        'acceptance_fraction': 0.1,
    }

    def __init__(self, initial_guess=None, emcee_settings=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_guess = initial_guess
        self._event = None

        # Default parameters_to_fit to initial_guess keys when not explicitly
        # provided via kwargs → MulensFitter.__init__.
        if self._parameters_to_fit is None and initial_guess is not None:
            self.parameters_to_fit = list(initial_guess.keys())

        # Merge user-supplied settings on top of class defaults.
        settings = dict(self.default_emcee_settings)
        if emcee_settings is not None:
            settings.update(emcee_settings)
        self.emcee_settings = settings

        # Register n_dim now if parameters_to_fit is already known.
        # Subclasses that finalise parameters_to_fit *after* super().__init__()
        # must update 'n_dim' themselves.
        if self._parameters_to_fit is not None:
            self.emcee_settings.setdefault('n_dim', len(self._parameters_to_fit))

    # ------------------------------------------------------------------ #
    # Pickling support                                                     #
    # ------------------------------------------------------------------ #

    def __getstate__(self):
        """
        Return the object state for pickling with ``_event`` set to ``None``.

        ``MulensModel.Event`` objects cannot be pickled after ``get_chi2()``
        has been called.  Call :meth:`initialize_event` after unpickling to
        rebuild the event.

        Returns
        -------
        dict
            ``__dict__`` with ``_event`` replaced by ``None``.
        """
        state = self.__dict__.copy()
        state['_event'] = None
        return state

    def __setstate__(self, state):
        """
        Restore the object state after unpickling.

        Parameters
        ----------
        state : dict
            State dictionary produced by :meth:`__getstate__`.
        """
        self.__dict__.update(state)

    # ------------------------------------------------------------------ #
    # Parameter helpers                                                    #
    # ------------------------------------------------------------------ #

    def get_parameter_name(self, parameter):
        """
        Strip the ``log_`` prefix from a parameter name if present.

        Parameters
        ----------
        parameter : str
            Parameter name, e.g. ``'log_rho'`` or ``'t_E'``.

        Returns
        -------
        str
            Name with ``log_`` prefix removed, e.g. ``'rho'`` or ``'t_E'``.
        """
        if parameter.startswith('log_'):
            return parameter[4:]
        return parameter

    def make_emcee_vector_from_ModelParameters(self, parameters):
        """
        Convert a ``MulensModel.ModelParameters`` object to an emcee vector.

        Parameters with a ``log_`` prefix are converted to log10 space.

        Parameters
        ----------
        parameters : MulensModel.ModelParameters
            Model parameters to convert.

        Returns
        -------
        list
            Vector of length ``len(parameters_to_fit)``.

        See Also
        --------
        event.setter : Performs the inverse conversion.
        """
        vector = []
        for parameter in self.parameters_to_fit:
            key = self.get_parameter_name(parameter)
            value = getattr(parameters, key)
            if key != parameter:  # log_ prefix
                value = np.log10(value)
            vector.append(value)
        return vector

    # ------------------------------------------------------------------ #
    # Event initialisation                                                 #
    # ------------------------------------------------------------------ #

    def initialize_event(self):
        """
        Initialise the ``MulensModel.Event`` from :attr:`initial_guess`.

        Parameters with a ``log_`` prefix are converted from log10 to linear
        space before the model is constructed.  Coordinates and flux-fixing
        are applied via :attr:`model_config` and :attr:`event_config`.

        Raises
        ------
        AttributeError
            If :attr:`initial_guess` is ``None``.
        """
        if self.initial_guess is None:
            raise AttributeError(
                'initial_guess must be set before calling initialize_event().')

        params = {}
        for key, value in self.initial_guess.items():
            actual_key = self.get_parameter_name(key)
            params[actual_key] = 10. ** value if actual_key != key else value

        model = self.model_config.build(
            parameters=params,
            magnification_methods=self.mag_methods,
        )
        self._event = self.event_config.build(
            model=model,
            datasets=self.datasets,
        )

    # ------------------------------------------------------------------ #
    # Starting ensemble                                                    #
    # ------------------------------------------------------------------ #

    def make_starting_vector(self):
        """
        Build the emcee starting ensemble by Gaussian perturbation of
        :attr:`initial_guess`.

        Each walker is obtained by adding independent ``N(0, sigma)`` noise to
        every component of :attr:`initial_guess`.  Parameters absent from
        :attr:`sigmas` (or when :attr:`sigmas` is ``None``) are left at their
        nominal value for every walker.

        Returns
        -------
        list of list
            ``n_walkers`` parameter vectors of length ``n_dim``.
        """
        n_walkers = self.emcee_settings['n_walkers']
        starting_vector = []
        for _ in range(n_walkers):
            walker = []
            for param in self.parameters_to_fit:
                value = self.initial_guess[param]
                sigma = (self.sigmas or {}).get(param)
                if sigma is not None:
                    value = value + np.random.normal(0., sigma)
                walker.append(value)
            starting_vector.append(walker)
        return starting_vector

    # ------------------------------------------------------------------ #
    # Event property                                                       #
    # ------------------------------------------------------------------ #

    @property
    def event(self):
        """
        The current ``MulensModel.Event`` object.

        Returns ``None`` until :meth:`initialize_event` has been called
        (or after unpickling).
        """
        return self._event

    @event.setter
    def event(self, theta):
        """
        Update model parameters from an emcee parameter vector.

        Parameters with a ``log_`` prefix are converted from log10 to linear
        space before assignment.

        Parameters
        ----------
        theta : array-like
            Vector of length ``len(parameters_to_fit)``.

        Raises
        ------
        AttributeError
            If the event has not yet been initialised.

        See Also
        --------
        make_emcee_vector_from_ModelParameters : Performs the inverse
            conversion.
        """
        if self._event is None:
            raise AttributeError(
                'Event has not been created. Call initialize_event() first.')
        for parameter, value in zip(self.parameters_to_fit, theta):
            key = self.get_parameter_name(parameter)
            if key != parameter:  # log_ prefix
                value = 10. ** value
            self._event.model.parameters.__setattr__(key, value)

    # ------------------------------------------------------------------ #
    # Likelihood / prior / probability                                     #
    # ------------------------------------------------------------------ #

    def ln_like(self, theta):
        """
        Log-likelihood for the emcee sampler (``-0.5 * chi2``).

        If ``'temperature'`` is present in :attr:`emcee_settings`, chi2 is
        divided by the square of the temperature (simulated annealing).

        Parameters
        ----------
        theta : array-like

        Returns
        -------
        float
            Log-likelihood, or ``-np.inf`` if evaluation failed.

        Notes
        -----
        The bare ``except`` clause silently catches all exceptions; consider
        catching specific exceptions to ease debugging.
        """
        self.event = theta
        try:
            chi2 = self.event.get_chi2()
            if 'temperature' in self.emcee_settings:
                chi2 /= self.emcee_settings['temperature'] ** 2
        except:
            return -np.inf
        return -0.5 * chi2

    def ln_prior(self, theta):
        """
        Log-prior: flat with hard lower bounds on positive-definite parameters.

        Rejects models where ``t_E``, ``rho``, ``q``, or ``s`` are
        non-positive.

        Parameters
        ----------
        theta : array-like

        Returns
        -------
        float
            ``0.0`` if all priors are satisfied; ``np.inf`` otherwise.

        Notes
        -----
        Returns ``np.inf`` (rather than ``-np.inf``) for rejected models;
        this is handled correctly by :meth:`ln_prob`.
        """
        for key, value in zip(self.parameters_to_fit, theta):
            if key in ('t_E', 'rho', 'q', 's') and value <= 0.:
                return np.inf
        return 0.0

    def ln_prob(self, theta):
        """
        Log-probability: prior + likelihood.

        Returns ``-np.inf`` when the prior is violated, when the likelihood
        cannot be evaluated, or when the likelihood is NaN (e.g. negative
        source fluxes).

        Parameters
        ----------
        theta : array-like

        Returns
        -------
        float

        See Also
        --------
        ln_prior : Log-prior function.
        ln_like : Log-likelihood function.
        """
        ln_prior_ = self.ln_prior(theta)
        if not np.isfinite(ln_prior_):
            return -np.inf
        ln_like_ = self.ln_like(theta)
        if np.isnan(ln_like_):
            return -np.inf
        return ln_prior_ + ln_like_

    # ------------------------------------------------------------------ #
    # Runner                                                               #
    # ------------------------------------------------------------------ #

    def run(self, verbose=False):
        """
        Fit the model using emcee MCMC sampling.

        Calls :meth:`make_starting_vector` to build the ensemble (subclass
        implementations may initialise the event as a side-effect).  If the
        event is still uninitialised after that call, :meth:`initialize_event`
        is invoked.  Sampling aborts early if the mean acceptance fraction
        drops below ``emcee_settings['acceptance_fraction']``.

        On completion, sets :attr:`best` to the highest-probability
        post-burn sample plus ``'chi2'``.

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, prints 16th/50th/84th-percentile summaries and logs
            the acceptance fraction every 100 steps.  Default ``False``.
        """
        starting_vector = self.make_starting_vector()

        # Subclass make_starting_vector() may initialise the event as a
        # side-effect; only call initialize_event() when that has not happened.
        if self._event is None:
            self.initialize_event()

        if self.pool:
            ncpu = cpu_count()
            print("{0} CPUs".format(ncpu))
            os.environ["OMP_NUM_THREADS"] = "1"
            pool = Pool()
            self.sampler = emcee.EnsembleSampler(
                self.emcee_settings['n_walkers'],
                self.emcee_settings['n_dim'],
                self.ln_prob,
                pool=pool)
        else:
            self.sampler = emcee.EnsembleSampler(
                self.emcee_settings['n_walkers'],
                self.emcee_settings['n_dim'],
                self.ln_prob)

        try:
            for _ in self.sampler.sample(
                    starting_vector, iterations=self.emcee_settings['n_steps']):

                if (self.sampler.iteration % 100) != 0:
                    continue
                if self.emcee_settings.get('acceptance_fraction') is None:
                    continue

                mean_af = np.mean(self.sampler.acceptance_fraction)

                if verbose:
                    print(self.sampler.iteration,
                          '{0:.3f}'.format(mean_af),
                          self.sampler.acceptance_fraction)

                if mean_af < self.emcee_settings['acceptance_fraction']:
                    print('Acceptance fraction too low! Minimum set to:',
                          self.emcee_settings['acceptance_fraction'])
                    break

        finally:
            if self.pool:
                pool.close()
                pool.join()

        n_burn = self.emcee_settings['n_burn']
        n_dim = self.emcee_settings['n_dim']
        samples = self.sampler.chain[:, n_burn:, :].reshape((-1, n_dim))

        if verbose:
            percentiles = np.percentile(samples, [16, 50, 84], axis=0)
            print("Fitted parameters:")
            for i in range(n_dim):
                med = percentiles[1, i]
                print("${:.5f}^{{+{:.5f}}}_{{-{:.5f}}}$ &".format(
                    med, percentiles[2, i] - med, med - percentiles[0, i]))

        prob = self.sampler.lnprobability[:, n_burn:].reshape((-1))
        best_index = np.argmax(prob)
        self.best_theta = samples[best_index]
        self.event = self.best_theta

        self.best = self._event.model.parameters.parameters
        self.best['chi2'] = self._event.get_chi2()


class AnomalyFitter(EmceeLCFitter):
    """
    Emcee fitter with automatic sigma estimation for microlensing anomaly fitting.

    Extends :class:`EmceeLCFitter` by adding a :attr:`default_parameters_to_fit`
    and automatic :attr:`sigmas` computation when sigmas are not explicitly
    provided.

    Unlike :class:`EmceeLCFitter`, ``initial_guess`` is required and
    ``parameters_to_fit`` defaults to :attr:`default_parameters_to_fit` rather
    than ``initial_guess.keys()``.  Users providing a custom ``initial_guess``
    must therefore include values for every parameter in
    :attr:`default_parameters_to_fit`, or supply ``parameters_to_fit``
    explicitly to restrict or reorder the fitted set.

    Sigma estimation follows a three-tier priority:

    1. **Explicit** — if ``sigmas`` is provided it is used unchanged.

    2. **From** ``anomaly_lc_params`` — if ``sigmas`` is not provided but
       ``anomaly_lc_params`` is, then for ``t_0`` and ``t_E``::

           sigma_t0 = sigma_tE = 0.01 * anomaly_lc_params['dt']

       All other parameters fall through to tier 3.

    3. **Defaults from** ``initial_guess``::

           t_0   :  0.00001
           u_0   :  0.001 * |u_0|
           t_E   :  0.001 * |t_E|
           alpha :  0.001
           log_X :  0.0001    # equivalent to sigma_X ~ 0.02% of X

       Parameters not matching any recognised pattern are omitted from
       ``sigmas``; the corresponding walkers are not perturbed in that
       dimension by :meth:`EmceeLCFitter.make_starting_vector`.

    Class Attributes
    ----------------
    default_parameters_to_fit : list
        ``['t_0', 'u_0', 't_E', 'log_rho', 'log_s', 'log_q', 'alpha']``

    Parameters
    ----------
    initial_guess : dict
        Starting parameter values keyed by parameter name.  Required.  Must
        contain a value for every parameter in :attr:`default_parameters_to_fit`
        (or every parameter in ``parameters_to_fit`` if that is supplied
        explicitly).  Keys may carry a ``log_`` prefix for parameters sampled
        in log10 space.
    anomaly_lc_params : dict, optional
        Parameters describing the anomaly light curve, as returned by
        ``AnomalyPropertyEstimator.get_anomaly_lc_parameters()``.  Used only
        to derive sigmas for ``t_0`` and ``t_E`` via the ``'dt'`` key.  Has
        no effect on ``initial_guess`` and no effect when ``sigmas`` is already
        provided explicitly.
    emcee_settings : dict, optional
        Passed through to :class:`EmceeLCFitter`.

    Notes
    -----
    :attr:`sigmas` are never derived from ``anomaly_lc_params`` directly.
    The typical source for PSPL parameter sigmas is the result of a prior PSPL
    fit (e.g. ``pspl_results.sigmas``), passed as explicit ``sigmas``.

    Subclasses that build ``initial_guess`` lazily (e.g.
    :class:`AnomalyFitterEnsembleInitialization`) should pass
    ``initial_guess=None`` to suppress automatic sigma computation, handling
    it themselves once ``initial_guess`` is available.

    All other parameters (``datasets``, ``sigmas``, ``model_config``,
    ``event_config``, ``pool``, …) are inherited from :class:`EmceeLCFitter`
    and :class:`MulensFitter`.

    See Also
    --------
    EmceeLCFitter : Parent class providing the full emcee mechanism.
    AnomalyFitterEnsembleInitialization : Subclass using
        ``WidePlanetEnsembleInitializer`` for the starting ensemble.
    """

    default_parameters_to_fit = [
        't_0', 'u_0', 't_E', 'log_rho', 'log_s', 'log_q', 'alpha']

    def __init__(
            self, initial_guess, anomaly_lc_params=None,
            emcee_settings=None, **kwargs):
        super().__init__(
            initial_guess=initial_guess,
            emcee_settings=emcee_settings,
            **kwargs)
        if self.mag_methods is None:
            raise ValueError(
                'mag_methods must be provided for AnomalyFitter.')

        self.anomaly_lc_params = anomaly_lc_params

        # Use default_parameters_to_fit when not explicitly provided.
        # This overrides EmceeLCFitter's default of list(initial_guess.keys()),
        # so initial_guess must contain values for all default parameters.
        if 'parameters_to_fit' not in kwargs:
            self.parameters_to_fit = list(self.default_parameters_to_fit)
            # n_dim must reflect the updated parameters_to_fit.  Direct
            # assignment rather than setdefault ensures a stale value from
            # EmceeLCFitter (set from initial_guess.keys()) is overwritten.
            self.emcee_settings['n_dim'] = len(self.parameters_to_fit)

        # Compute sigmas only when not explicitly provided and initial_guess
        # is available.  The guard on initial_guess allows subclasses that
        # build initial_guess lazily to pass None and defer sigma computation.
        if self.sigmas is None and self.initial_guess is not None:
            self.sigmas = self._compute_sigmas()

    def _compute_sigmas(self):
        """
        Compute sigmas following the three-tier priority.

        Starts from :meth:`_default_sigmas`, then overrides ``t_0`` and
        ``t_E`` from ``anomaly_lc_params['dt']`` if ``anomaly_lc_params`` is
        set.

        Returns
        -------
        dict
            Sigma for each recognised parameter in :attr:`parameters_to_fit`.
        """
        sigmas = self._default_sigmas()

        if self.anomaly_lc_params is not None:
            dt = self.anomaly_lc_params['dt']
            if 't_0' in self.parameters_to_fit:
                sigmas['t_0'] = 0.01 * dt
            if 't_E' in self.parameters_to_fit:
                sigmas['t_E'] = 0.01 * dt

        return sigmas

    def _default_sigmas(self):
        """
        Compute default sigmas from :attr:`initial_guess`.

        For ``log_X`` parameters the step size is derived so that the
        equivalent linear-space step equals 0.01% of the parameter value:

        .. math::

            \\sigma_{\\log X} = \\frac{\\sigma_X}{X \\ln 10}
                              = \\frac{0.0001}{\\ln 10}

        Parameters not matching any recognised pattern are omitted.

        Returns
        -------
        dict
            Default sigma for each recognised parameter in
            :attr:`parameters_to_fit`.
        """
        sigmas = {}
        for param in self.parameters_to_fit:
            if param == 't_0':
                sigmas[param] = 0.00001
            elif param == 'u_0':
                sigmas[param] = 0.001 * abs(self.initial_guess['u_0'])
            elif param == 't_E':
                sigmas[param] = 0.001 * abs(self.initial_guess['t_E'])
            elif param == 'alpha':
                sigmas[param] = 0.001
            elif param.startswith('log_'):
                sigmas[param] = 0.0001

        return sigmas


class WidePlanetEnsembleInitialization(AnomalyFitter):
    """
    Anomaly fitter using ``WidePlanetEnsembleInitializer`` for the starting
    ensemble.

    Extends :class:`AnomalyFitter` for the wide-planet geometry.  Overrides
    :meth:`make_starting_vector` to drive ``WidePlanetEnsembleInitializer`` and
    :meth:`initialize_event` to use the resulting ``initial_model`` and
    ``mag_methods``.

    :attr:`default_parameters_to_fit` is inherited from :class:`AnomalyFitter`.
    ``parameters_to_fit`` and ``'n_dim'`` are finalised in the parent
    ``__init__``; this class adds no further changes to either.

    ``initial_guess`` is built lazily as a side-effect of the first call to
    :meth:`make_starting_vector`; ``None`` is passed to the parent to suppress
    automatic sigma computation.  Sigmas passed explicitly (typically PSPL fit
    sigmas) are forwarded to ``WidePlanetEnsembleInitializer`` for PSPL
    perturbation; when none are provided they default to ``{}`` (no
    perturbation).

    Parameters
    ----------
    anomaly_lc_params : dict, optional
        Passed to ``WidePlanetEnsembleInitializer``.  Also inspected by the
        parent :meth:`AnomalyFitter._compute_sigmas` — but since
        ``initial_guess`` is ``None`` at construction time, automatic sigma
        computation is suppressed and ``anomaly_lc_params`` is used only by
        the initialiser itself.
    emcee_settings : dict, optional
        Passed through to :class:`AnomalyFitter`.

    Notes
    -----
    ``initial_model`` and ``mag_methods`` are populated lazily as side-effects
    of the first call to :meth:`make_starting_vector` (triggered by
    :meth:`~EmceeLCFitter.run` or by accessing either property directly).

    All other parameters are inherited from :class:`AnomalyFitter` and
    :class:`MulensFitter`.

    See Also
    --------
    AnomalyFitter : Parent class providing default parameters and sigma tiers.
    EmceeLCFitter : Grandparent providing the full emcee run loop.
    WidePlanetEnsembleInitializer : Builds the starting ensemble.
    """

    def __init__(self, anomaly_lc_params=None, emcee_settings=None, **kwargs):
        # Pass initial_guess=None to suppress automatic sigma computation;
        # initial_guess is built lazily inside make_starting_vector().
        super().__init__(
            initial_guess=None,
            anomaly_lc_params=anomaly_lc_params,
            emcee_settings=emcee_settings,
            **kwargs)

        # Sigmas are forwarded to WidePlanetEnsembleInitializer for PSPL
        # perturbation.  Default to {} (no perturbation) when not provided;
        # automatic sigma computation was suppressed above because
        # initial_guess=None.
        if self.sigmas is None:
            self.sigmas = {}

        self._initializer = None
        self._starting_vector = None
        self._initial_model = None
        self._mag_methods = None

    # ------------------------------------------------------------------ #
    # Lazy properties populated by make_starting_vector()                 #
    # ------------------------------------------------------------------ #

    @property
    def initial_model(self):
        """
        Initial model parameter dict; populated by :meth:`make_starting_vector`.

        Accessing this property before :meth:`make_starting_vector` has been
        called triggers full ensemble initialisation as a side-effect.
        """
        if self._initial_model is None:
            self.make_starting_vector()
        return self._initial_model

    @initial_model.setter
    def initial_model(self, value):
        self._initial_model = value

    @property
    def mag_methods(self):
        """
        Magnification-methods list; populated by :meth:`make_starting_vector`.

        Accessing this property before :meth:`make_starting_vector` has been
        called triggers full ensemble initialisation as a side-effect.
        """
        if self._mag_methods is None:
            self.make_starting_vector()
        return self._mag_methods

    @mag_methods.setter
    def mag_methods(self, value):
        self._mag_methods = value

    # ------------------------------------------------------------------ #
    # Overridden starting ensemble                                         #
    # ------------------------------------------------------------------ #

    def make_starting_vector(self):
        """
        Build the starting ensemble using ``WidePlanetEnsembleInitializer``.

        Runs ``n_walkers`` ``WidePlanetGridSearchEstimators`` with perturbed
        PSPL parameters, sorts results by chi2, and converts the best
        ``n_walkers`` rows to emcee parameter vectors.  Sets
        :attr:`initial_model` and :attr:`mag_methods` as side-effects and
        initialises the event if not already created.  Subsequent calls
        return the cached result without re-running the initialiser.

        Returns
        -------
        list of list
            ``n_walkers`` parameter vectors of length ``n_dim``.
        """
        if self._starting_vector is not None:
            return self._starting_vector

        self._initializer = WidePlanetEnsembleInitializer(
            datasets=self.datasets,
            anomaly_params=self.anomaly_lc_params,
            sigmas=self.sigmas,
            n_estimators=self.emcee_settings['n_walkers'],
            pspl_chi2=getattr(self, 'pspl_chi2', None),
        )

        # Store via setters so that the lazy properties resolve on subsequent
        # accesses without triggering another initialiser run.
        self.initial_model = self._initializer.initial_model
        self.mag_methods = self._initializer.mag_methods

        if self._event is None:
            self.initialize_event()

        df = self._initializer.results.sort_values('chi2')
        top_rows = df.head(self.emcee_settings['n_walkers'])

        starting_vector = []
        for _, row in top_rows.iterrows():
            params = {k: row[k] for k in
                      ['t_0', 'u_0', 't_E', 's', 'q', 'rho', 'alpha']}
            vector = self.make_emcee_vector_from_ModelParameters(
                MulensModel.ModelParameters(params))
            starting_vector.append(vector)

        self._starting_vector = starting_vector
        return starting_vector

    # ------------------------------------------------------------------ #
    # Overridden event initialisation                                      #
    # ------------------------------------------------------------------ #

    def initialize_event(self):
        """
        Initialise the ``MulensModel.Event`` from :attr:`initial_model` and
        :attr:`mag_methods`.

        Both are set as side-effects of :meth:`make_starting_vector`.
        ``default_magnification_method`` is set to
        ``'point_source_point_lens'`` to cover regions outside the explicitly
        specified method windows.

        Raises
        ------
        AttributeError
            If :attr:`initial_model` or :attr:`mag_methods` has not been set
            (i.e. :meth:`make_starting_vector` has not been called yet).
        """
        if self._initial_model is None:
            raise AttributeError(
                'initial_model is not set. Call make_starting_vector() first.')
        if self._mag_methods is None:
            raise AttributeError(
                'mag_methods is not set. Call make_starting_vector() first.')

        model = self.model_config.build(
            parameters=self._initial_model,
            magnification_methods=self._mag_methods,
            default_magnification_method='point_source_point_lens',
        )
        self._event = self.event_config.build(
            model=model,
            datasets=self.datasets,
        )
