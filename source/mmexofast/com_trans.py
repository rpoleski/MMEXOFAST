"""
Perform coordinate transformations between different microlensing coordinate systems.
"""

import numpy as np
import warnings


def _get_co_mass_co_manif_offset(params):
    """
    Calculate the offset in t_0 and u_0 when transforming between the center
    of mass and center of magnification frames, taking into account the angle
    of the source trajectory. Only relevant when s > 1; when s <= 1 the two
    frames are identical and (0., 0.) is returned.

    Parameters
    ----------
    params : dict
        Required keys: 's', 'q', 'alpha', 't_E'

    Returns
    -------
    delta_t0 : float
        Change in t_0 between the two frames
    delta_u0 : float
        Change in u_0 between the two frames
    """
    if params['s'] > 1.:
        delta_x = ((params['q'] / (1. + params['q'])) *
                   ((1. / params['s']) - params['s']))
        delta_u0 = delta_x * np.sin(np.deg2rad(params['alpha']))
        delta_t0 = delta_x * params['t_E'] * np.cos(
            np.deg2rad(params['alpha']))
    else:
        delta_t0 = 0.
        delta_u0 = 0.

    return delta_t0, delta_u0


def co_mass_to_co_magnif(params):
    """
    Transform t_0 and u_0 from the center of mass to the center of magnification frame.

    Parameters
    ----------
    params : dict
        Required keys: 't_0', 'u_0', 's', 'q', 'alpha', 't_E'

    Returns
    -------
    dict
        Keys 't_0' and 'u_0' in the center of magnification frame.
    """
    delta_t0, delta_u0 = _get_co_mass_co_manif_offset(params)
    return {'t_0': params['t_0'] - delta_t0,
            'u_0': params['u_0'] - delta_u0}


def co_magnif_to_co_mass(params):
    """
    Transform t_0 and u_0 from the center of magnification to the center of mass frame.

    Parameters
    ----------
    params : dict
        Required keys: 't_0', 'u_0', 's', 'q', 'alpha', 't_E'

    Returns
    -------
    dict
        Keys 't_0' and 'u_0' in the center of mass frame.
    """
    delta_t0, delta_u0 = _get_co_mass_co_manif_offset(params)

    return {'t_0': params['t_0'] + delta_t0,
            'u_0': params['u_0'] + delta_u0}


def primary_to_co_magnif(params):
    """
    Transform t_0 and u_0 from the primary location to the center of magnification frame.

    Parameters
    ----------
    params : dict
        Required keys: 't_0', 'u_0', 's', 'q', 'alpha', 't_E'

    Returns
    -------
    dict
        Keys 't_0' and 'u_0' in the center of magnification frame.

    Warnings
    --------
    This function is untested.
    """
    warnings.warn("primary_to_co_magnif is untested.", UserWarning)

    delta_x = params['q'] * ((1. / params['s']) + params['s'])
    delta_u0 = delta_x * np.sin(np.deg2rad(params['alpha']))
    delta_t0 = delta_x * params['t_E'] * np.cos(
        np.deg2rad(-params['alpha']))

    return {'t_0': params['t_0'] - delta_t0,
            'u_0': params['u_0'] + delta_u0}


def primary_to_co_mass(params):
    """
    Transform t_0 and u_0 from the primary location to the center of mass frame.

    Parameters
    ----------
    params : dict
        Required keys: 't_0', 'u_0', 's', 'q', 'alpha', 't_E'

    Returns
    -------
    dict
        Keys 't_0' and 'u_0' in the center of mass frame.

    Warnings
    --------
    This function is untested.
    """
    warnings.warn("primary_to_co_magnif is untested.", UserWarning)

    new_coords = primary_to_co_magnif(params)
    new_params = {'t_0': new_coords['t_0'], 'u_0': new_coords['u_0'], 't_E': params['t_E'],
                  's': params['s'], 'q': params['q'], 'alpha': params['alpha']}

    return co_magnif_to_co_mass(new_params)