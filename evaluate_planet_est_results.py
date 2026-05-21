import glob
import ast
import os.path
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from examples.use_cases.DC18_classes import DC18Answers


def get_results():
    results = []
    log_files = glob.glob('temp_output/W*/WFIRST*.log')
    for log_file in log_files:
        print(log_file)
        with open(log_file) as f:
            for line in reversed(f.readlines()):
                if line.startswith("Estimated binary params"):
                    dict_str = line.split(": ", 1)[1]  # Split on first ": " only
                    dict_str = re.sub(r'np\.float64\(([^)]+)\)', r'\1', dict_str)
                    params = ast.literal_eval(dict_str)
                    params['idx'] = int(os.path.basename(log_file).split('.')[1]) - 1
                    # idx = lc_num - 1
                    results.append(params)
                    break

    return pd.DataFrame(results)


class EvaluateResults():

    def __init__(self):
        self.results = get_results()
        truth = DC18Answers()
        #print(truth.print_wide_orbit_planets())
        #raise NotImplementedError('DC18Answers does not parse columns correctly.')
        # Default suffixes: _x (results) and _y (truth)
        self.results = self.results.merge(
            truth.data.add_suffix('_true'),
            left_on='idx',
            right_index=True,
            how='left',  # Keep all rows in self.results
        )
        print_columns = ['idx', 't_0', 't0_true', 'u_0', 'u0_true', 't_E', 'tE_true', 'rho', 'rhos_true',
                         's', 's_true', 'q', 'q_true', 'alpha', 'alpha_true']

        print(self.results[print_columns].sort_values('idx'))

    def _make_scatter_plot(self, key, log=False):
        if key == 'rho':
            truth_key = key + 's'
        else:
            truth_key = key.replace('_', '')

        truth_key += '_true'
        fit_value = self.results[key]
        true_value = self.results[truth_key]
        if (key == 'u_0') and log:
            fit_value = np.abs(fit_value)
            true_value = np.abs(true_value)

        if log:
            delta = np.log10(fit_value) - np.log10(true_value)
            value = np.log10(true_value)
            xlabel = f'log {key}_true'
            ylabel = f'Delta log {key}'
        else:
            delta = fit_value - true_value
            value = true_value
            xlabel = f'{key}_true'
            ylabel = f'Delta {key}'

        colors = np.where(self.results['s_true'] < 1, 'darkorange', 'darkcyan')
        facecolors = np.where(self.results['q_true'] < 0.03, colors, 'none')
        low_mag = self.results['u0_true'] > 0.05

        for mask, marker in zip([low_mag, ~low_mag], ['o', '^']):
            plt.scatter(
                value[mask], delta[mask] / value[mask],
                c=colors[mask], facecolors=facecolors[mask], marker=marker)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.minorticks_on()
        plt.tight_layout()

    def make_all_plots(self):
        plot_list = {'u_0': True, 't_E': False, 'rho': True, 'alpha': False, 's': True, 'q': True}
        for key, log in plot_list.items():
            plt.figure()
            self._make_scatter_plot(key, log=log)


if __name__=='__main__':
    evaluator = EvaluateResults()
    evaluator.make_all_plots()
    plt.show()
