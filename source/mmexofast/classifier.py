import numpy as np


class AnomalyClassifier(object):
    """
    Classifies a microlensing event anomaly as 'close', 'wide', or 'high_mag'.

    Uses lightcurve and anomaly parameters from
    AnomalyPropertyEstimator.get_anomaly_lc_parameters() to determine
    the classification of the event.
    """

    def __init__(self):
        pass

    def classify(self, params):
        """
        Use the lightcurve and anomaly properties to determine what kind of fit is needed.

        Parameters
        ----------
        params : dict
            Results of AnomalyPropertyEstimator.get_anomaly_lc_parameters()

        Returns
        -------
        str
            One of 'close', 'wide', 'high_mag'
        """
        if np.abs(params['u_0']) < 0.01:
            return 'high_mag'

        if params['dmag'] < 0:
            if np.abs(params['u_0']) > 0.05:
                return 'wide'
            else:
                return 'high_mag'

        if params['dmag'] > 0:
            return 'close'
