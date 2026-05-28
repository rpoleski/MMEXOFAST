from __future__ import annotations

import dataclasses
from typing import Optional

import MulensModel

@dataclasses.dataclass
class ModelConfig:
    """
    Packages all arguments needed to construct and configure a
    ``MulensModel.Model``.

    Parameters that depend on the specific model type —
    ``magnification_methods``, ``default_magnification_method``, and
    ``magnification_methods_parameters`` — are passed directly to
    ``build()`` rather than stored here, because they vary per model type
    (e.g. point source vs. binary lens) while these fields are constant
    across all models in a workflow.

    Parameters
    ----------
    coords : str or MulensModel.Coordinates, optional
        Sky coordinates of the event.
    ra : float, optional
        Right ascension in degrees.
    dec : float, optional
        Declination in degrees.
    ephemerides_file : str, optional
        Path to ephemerides file for satellite datasets.
    limb_coeff_gamma : dict, optional
        Gamma limb-darkening coefficients keyed by bandpass.
    limb_coeff_u : dict, optional
        Linear limb-darkening coefficients keyed by bandpass.
    """

    coords: Optional[object] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    ephemerides_file: Optional[str] = None
    limb_coeff_gamma: Optional[dict] = None
    limb_coeff_u: Optional[dict] = None

    def build(
        self,
        parameters: dict,
        magnification_methods=None,
        magnification_methods_parameters=None,
        default_magnification_method=None,
    ) -> MulensModel.Model:
        """
        Create and fully configure a ``MulensModel.Model``.

        Parameters
        ----------
        parameters : dict
            The model parameters for this particular instance.
        magnification_methods : list, optional
            Magnification methods in MulensModel convention. Varies by
            model type; not stored on the config.
        magnification_methods_parameters : dict, optional
            Parameters for the magnification methods.
        default_magnification_method : str, optional
            Default magnification method to use outside the ranges
            specified in ``magnification_methods``.

        Returns
        -------
        MulensModel.Model
        """
        model = MulensModel.Model(
            parameters=parameters, **self._constructor_kwargs()
        )
        self._apply_post_construction(
            model,
            magnification_methods=magnification_methods,
            magnification_methods_parameters=magnification_methods_parameters,
            default_magnification_method=default_magnification_method,
        )
        return model

    def _constructor_kwargs(self) -> dict:
        """
        Return keyword arguments for ``MulensModel.Model.__init__``.

        Returns
        -------
        dict
            Only includes fields that are not None.
        """
        constructor_fields = ['coords', 'ra', 'dec', 'ephemerides_file']
        return {
            f: getattr(self, f)
            for f in constructor_fields
            if getattr(self, f) is not None
        }

    def _apply_post_construction(
        self,
        model: MulensModel.Model,
        magnification_methods=None,
        magnification_methods_parameters=None,
        default_magnification_method=None,
    ) -> None:
        """
        Apply post-construction settings to a ``MulensModel.Model``.

        Parameters
        ----------
        model : MulensModel.Model
            The model to configure.
        magnification_methods : list, optional
        magnification_methods_parameters : dict, optional
        default_magnification_method : str, optional
        """
        if magnification_methods is not None:
            model.set_magnification_methods(magnification_methods)
        if magnification_methods_parameters is not None:
            model.set_magnification_methods_parameters(
                magnification_methods_parameters
            )
        if default_magnification_method is not None:
            model.default_magnification_method = default_magnification_method
        if self.limb_coeff_gamma is not None:
            for band, value in self.limb_coeff_gamma.items():
                model.set_limb_coeff_gamma(value, band)
        if self.limb_coeff_u is not None:
            for band, value in self.limb_coeff_u.items():
                model.set_limb_coeff_u(value, band)


@dataclasses.dataclass
class EventConfig:
    """
    Packages all arguments needed to construct a ``MulensModel.Event``.

    ``model`` and ``datasets`` are always passed directly to ``build()``
    because ``model`` is produced by ``ModelConfig.build()`` and
    ``datasets`` are managed externally.

    Parameters
    ----------
    coords : str or MulensModel.Coordinates, optional
        Sky coordinates of the event.
    fix_blend_flux : dict, optional
        Mapping of ``MulensModel.MulensData`` to blend flux fixing value.
    fix_source_flux : dict, optional
        Mapping of ``MulensModel.MulensData`` to source flux fixing value.
    fix_source_flux_ratio : dict, optional
        Mapping of ``MulensModel.MulensData`` to source flux ratio fixing
        value.
    data_ref : int, optional
        Index of the reference dataset. Defaults to 0.
    """

    coords: Optional[object] = None
    fix_blend_flux: Optional[dict] = None
    fix_source_flux: Optional[dict] = None
    fix_source_flux_ratio: Optional[dict] = None
    data_ref: int = 0

    def build(
        self,
        model: MulensModel.Model,
        datasets: list,
    ) -> MulensModel.Event:
        """
        Create a ``MulensModel.Event``.

        Parameters
        ----------
        model : MulensModel.Model
            Already-constructed model; use ``ModelConfig.build()``.
        datasets : list
            List of ``MulensModel.MulensData`` objects.

        Returns
        -------
        MulensModel.Event
        """
        return MulensModel.Event(
            model=model, datasets=datasets, **self._constructor_kwargs()
        )

    def _constructor_kwargs(self) -> dict:
        """
        Return keyword arguments for ``MulensModel.Event.__init__``.

        Returns
        -------
        dict
            Only includes fields that are not None or non-default.
        """
        constructor_fields = [
            'coords', 'fix_blend_flux', 'fix_source_flux',
            'fix_source_flux_ratio',
        ]
        kwargs = {
            f: getattr(self, f)
            for f in constructor_fields
            if getattr(self, f) is not None
        }
        if self.data_ref != 0:
            kwargs['data_ref'] = self.data_ref
        return kwargs
