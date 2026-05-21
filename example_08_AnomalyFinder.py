"""
Example to show that the AnomalyFinder works as expected.
"""
import numpy as np
import matplotlib.pyplot as plt
import os.path
import re, ast

import MulensModel
import exozippy as mmexo
from DC18_classes import TestDataSet


class AnomalyFinderTest:
    """
    Tests and demonstrates the AnomalyFinder grid search functionality
    for a given light curve number.
    """

    def __init__(self, lc_num):
        self.lc_num = lc_num
        self._pspl_params = None
        self._fitter = None
        self._datasets = None
        self._t_start = None
        self._t_stop = None
        self.af = None

    @property
    def pspl_params(self):
        if self._pspl_params is None:
            log_file = os.path.join(
                mmexo.MODULE_PATH, 'EXOZIPPy', 'DC18Test', 'temp_output',
                f'W{self.lc_num:03}', f'WFIRST.{self.lc_num:03}.log')
            with open(log_file) as f:
                for line in reversed(f.readlines()):
                    if line.startswith("Static PSPL"):
                        dict_str = line.split(": ", 1)[1]
                        dict_str = re.sub(r'np\.float64\(([^)]+)\)', r'\1', dict_str)
                        params_dict = ast.literal_eval(dict_str)
                        params_dict.pop('chi2')
                        self._pspl_params = params_dict
        return self._pspl_params

    @property
    def fitter(self):
        if self._fitter is None:
            test_data = TestDataSet(lc_num=self.lc_num)
            self._fitter = mmexo.mmexofast.MMEXOFASTFitter(
                files=[test_data.file_w149, test_data.file_z087],
                initial_results={'PSPL static': {'params': self.pspl_params}})
        return self._fitter

    @property
    def datasets(self):
        if self._datasets is None:
            self._datasets = self.fitter.datasets
        return self._datasets

    def _compute_t_range(self):
        # Single computation shared by t_start and t_stop
        data = self.datasets[0]
        t_0 = self.pspl_params['t_0']
        times = data.time[np.abs(data.time - t_0) < 180.]
        self._t_start = np.nanmin(times)
        self._t_stop = np.nanmax(times)

    @property
    def t_start(self):
        if self._t_start is None:
            self._compute_t_range()
        return self._t_start

    @property
    def t_stop(self):
        if self._t_stop is None:
            self._compute_t_range()
        return self._t_stop

    def run_grid_search(self):
        self.fitter.pspl_params = self.pspl_params
        self.fitter.compute_residuals()
        self.af = mmexo.gridsearches.AnomalyFinderGridSearch(
            residuals=self.fitter.residuals,
            t_0_min=self.t_start,
            t_0_max=self.t_stop)
        
        #plt.figure()
        #for dataset in self.fitter.residuals:
        #    dataset.plot()
        #
        #plt.show()

        self.af.run(verbose=True)
        print('Best:')
        print(self.af.best)
        print('# of anomalies', len(self.af.anomalies))

    def plot_efs_fit_function(self, test_params, verbose=False):
        datasets = self.af.get_trimmed_datasets(test_params)
        model = mmexo.gridsearches.EFSFitFunction(datasets, test_params)
        model.update_all()
        theta_new = model.theta + model.get_step()
        model.update_all(theta=theta_new)
        if verbose:
            print('chi2', model.chi2)

        plt.errorbar(
            datasets[0].time, datasets[0].flux, yerr=datasets[0].err_flux,
            fmt='o')
        plt.axhline(0, color='black', linestyle='--')
        plt.plot(
            model.data[model.data_indices[0]:model.data_indices[1], 0],
            model.ymod[model.data_indices[0]:model.data_indices[1]],
            color='black', zorder=5)
        plt.xlabel('HJD')
        plt.ylabel('W149 flux')
        plt.minorticks_on()
        plt.tight_layout()

    def plot_single_fit(self):
        plt.figure()
        plt.title('Test Single element')
        for anomaly in self.af.anomalies[:2]:
            #print(anomaly)
            test_params = {'t_0': anomaly[0], 't_eff': anomaly[1], 'j': anomaly[2]}
            #print(test_params)
            trimmed_residuals = self.af.get_trimmed_datasets(test_params)
            print('chi2_zero', np.sum(np.hstack(
                [(dataset.flux / dataset.err_flux)**2
                 for dataset in trimmed_residuals])))
            self.plot_efs_fit_function(test_params, verbose=True)

    def plot_chi2_grid(self):
        labels = ['1', '2', 'flat', 'zero']
        for j in range(4):
            sorted_idx = np.argsort(self.af.results[:, j])[::-1]
            plt.figure()
            plt.scatter(
                self.af.grid_t_0[sorted_idx], self.af.grid_t_eff[sorted_idx],
                c=self.af.results[sorted_idx, j],
                edgecolors='black', cmap='tab20b')
            plt.colorbar(label='chi2_{0}'.format(labels[j]))
            plt.scatter(
                self.af.best['t_0'], self.af.best['t_eff'],
                color='black', marker='x', zorder=10, s=100)
            plt.title('chi2_{0}'.format(labels[j]))
            plt.minorticks_on()
            plt.xlabel('t_0')
            plt.ylabel('t_eff')
            plt.yscale('log')
            plt.tight_layout()

        plt.figure(figsize=(8, 4))
        for j in [1, 2]:
            plt.subplot(1, 2, j)
            plt.title('j={0}'.format(j))
            dchi2_zero = self.af.results[:, 3] - self.af.results[:, j - 1]
            sorted_idx = np.argsort(dchi2_zero)
            plt.scatter(
                self.af.grid_t_0[sorted_idx], self.af.grid_t_eff[sorted_idx],
                c=dchi2_zero[sorted_idx],
                edgecolors='black', cmap='tab20b', vmin=0)
            plt.colorbar(label='chi2_zero - chi2')
            plt.scatter(
                self.af.best['t_0'], self.af.best['t_eff'],
                color='black', marker='x', zorder=10)
            plt.minorticks_on()
            plt.xlabel('t_0')
            plt.ylabel('t_eff')
            plt.yscale('log')
            plt.tight_layout()

    def plot_event(self):
        plt.figure()
        plt.title('Event')
        event = MulensModel.Event(
            datasets=self.datasets,
            model=MulensModel.Model(self.pspl_params))
        event.plot_data(phot_fmt='flux')
        plt.gca().invert_yaxis()

        times = np.linspace(self.t_start, self.t_stop, 1000)
        ref_fluxes = event.get_ref_fluxes()
        print(ref_fluxes)
        model_fluxes = (ref_fluxes[0][0] * event.model.get_magnification(times) +
                        ref_fluxes[1])
        print(model_fluxes.shape)
        plt.plot(times, model_fluxes, color='black', zorder=5)

        plt.axvline(self.af.best['t_0'], color='black')
        plt.axvline(self.af.best['t_0'] - self.af.best['t_eff'],
                    color='black', linestyle='--')
        plt.axvline(self.af.best['t_0'] + self.af.best['t_eff'],
                    color='black', linestyle='--')
        plt.xlabel('HJD')
        plt.ylabel('W149 flux')
        plt.minorticks_on()
        plt.tight_layout()

    def plot_residuals(self):
        plt.figure()
        plt.title('Residuals')
        self.plot_efs_fit_function(self.af.best)

    def plot_all(self):
        self.plot_single_fit()
        self.plot_chi2_grid()
        self.plot_event()
        self.plot_residuals()
        plt.show()

    def run(self):
        self.run_grid_search()
        self.plot_all()


if __name__ == '__main__':
    lc_num = 92
    test = AnomalyFinderTest(lc_num)
    test.run()