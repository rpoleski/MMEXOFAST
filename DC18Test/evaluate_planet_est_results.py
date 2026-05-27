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
    log_files = glob.glob('temp_output/no_par/W*/WFIRST.*.log')
    #log_files = glob.glob('temp_output/W004/WFIRST.004.log')
    failed = []
    hm = []
    print('\nTotal Fits:', len(log_files), '\n')
    for log_file in log_files:
        lc_num = int(os.path.basename(log_file).split('.')[1])
        print(lc_num, log_file)
        with open(log_file) as f:
            for line in reversed(f.readlines()):
                #print(line)
                #print(line.startswith("Estimated binary params"))
                if line.startswith("Estimated binary params"):
                    dict_str = line.split(": ", 1)[1]  # Split on first ": " only
                    dict_str = re.sub(r'np\.float64\(([^)]+)\)', r'\1', dict_str)
                    #print(dict_str)
                    params = ast.literal_eval(dict_str)
                    params['idx'] = lc_num - 1
                    params['t_0'] -= 2458234.
                    # idx = lc_num - 1
                    results.append(params)
                    break

                if line.strip().endswith("high_mag"):
                    hm.append(lc_num)
                    break
            else:
                failed.append(lc_num)

    print('\nclassified as hm:', sorted(hm))
    print('Total: ', len(hm), '\n')
    print('\nest_binary_params failed: ', sorted(failed), '\nTotal: ', len(failed), '\n')
    return pd.DataFrame(results)


class EvaluateResults():
    eval_columns = ['idx', 'u0_true', 'u_0', 'tE_true', 't_E', 's_true', 's', 'q_true', 'q']
    print_columns = ['idx', 't_0', 't0_true', 'u_0', 'u0_true', 't_E', 'tE_true', 'rho', 'rhos_true',
                     's', 's_true', 'q', 'q_true', 'alpha', 'alpha_true']

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

        print('\nsucceeded:', len(self.results))
        with pd.option_context('display.width', None):
              print(self.results[self.print_columns].sort_values('idx'))

    def _get_truth_key(self, key):
        if key == 'rho':
            truth_key = key + 's'
        else:
            truth_key = key.replace('_', '')

        truth_key += '_true'
        return truth_key

    def _plot(self, x, y, ylim=None, log=False):
        plt.figure()
        
        colors = np.where(self.results['s_true'] < 1, 'darkorange', 'darkcyan')
        facecolors = np.where(self.results['q_true'] < 0.03, colors, 'none')
        low_mag = self.results['u0_true'].abs() > 0.05

        for mask, marker in zip([low_mag, ~low_mag], ['o', 'd']):
            plt.scatter(
                x[mask], y[mask],
                c=colors[mask], facecolors=facecolors[mask], marker=marker,
                clip_on=False, zorder=3, )

        if log:
            # Define masks for each out-of-range direction
            mask_above = y > ylim[1]
            mask_below = y < ylim[0]

            # Plot out-of-range points as triangles at the plot edge
            plt.scatter(x[mask_above], np.full(mask_above.sum(), ylim[1]),
                        marker='^', c=colors[mask_above], facecolors=facecolors[mask_above],
                        clip_on=False, zorder=3, )
            plt.scatter(x[mask_below], np.full(mask_below.sum(), ylim[0]),
                        marker='v', c=colors[mask_below], facecolors=facecolors[mask_below],
                        clip_on=False, zorder=3, )

    def _make_scatter_plot(self, key, log=False):
        truth_key = self._get_truth_key(key)

        fit_value = self.results[key]
        true_value = self.results[truth_key]

        if (key == 'u_0') and log:
            fit_value = np.abs(fit_value)
            true_value = np.abs(true_value)

        if log:
            delta = np.log10(fit_value) - np.log10(true_value)
            value = np.log10(true_value)
            xlabel = f'log ({key}_true)'
            ylabel = f'log({key}) - log(True)'
            ylim = (-1.5, 1.5)

        else:
            value = true_value
            delta = (fit_value - true_value)
            xlabel = f'{key}_true'
            ylabel = f'({key} - True)'
            ylim = None

        self._plot(value, delta, ylim=ylim, log=log)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(ylim)
        plt.minorticks_on()
        plt.tight_layout()

    def make_all_delta_plots(self):
        plot_list = {'u_0': True, 't_E': False, 'rho': True, 'alpha': False, 's': True, 'q': True}
        for key, log in plot_list.items():
            plt.figure()
            self._make_scatter_plot(key, log=log)
            plt.savefig(f'temp_output/no_par/figs/{key}_deltas.png', dpi=300)

    def make_scatter_plot(self, key, log=False):
        fit_value = self.results[key]
        true_value = self.results[self._get_truth_key(key)]

        self._plot(true_value, fit_value)
        
        plt.gca().set_aspect('equal')
        ylim = plt.gca().get_ylim()
        xlim = plt.gca().get_xlim()
        min_lim = np.min([xlim[0], ylim[0]])
        max_lim = np.max([xlim[1], ylim[1]])
        plt.plot([min_lim, max_lim], [min_lim, max_lim], zorder=0, color='black', clip_on=True)

        if log:
            plt.xlabel(f'log {key} (True)')
            plt.xscale('log')
            plt.ylabel(f'log {key} (Fitted)')
            plt.yscale('log')
        else:
            plt.xlabel(f'{key} (True)')
            plt.ylabel(f'{key} (Fitted)')

        plt.minorticks_on()
        plt.tight_layout()

    def make_all_scatter_plots(self):
        for key in ['u_0', 't_E', 'rho', 'alpha', 's', 'q']:
            if key == 'q' or key == 'rho':
                log = True
            else:
                log = False

            self.make_scatter_plot(key, log=log)
            plt.savefig(f'temp_output/no_par/figs/{key}_vs.png', dpi=300)

    def is_log_q_good(self, threshold=0.5):
        delta = np.log10(self.results['q']) - np.log10(self.results['q_true'])
        good = np.abs(delta) < threshold
        print(f'|dlog q|< {threshold}: {np.sum(good)}')
        print('good:', (self.results[good]['idx']+1).tolist())
        print('bad:', (self.results[~good]['idx'] + 1).tolist())
        with pd.option_context('display.width', None, 'display.max_rows', None):
            print('\ngood:\n', self.results[good][self.eval_columns].sort_values('idx'))
            print('\nbad:\n', self.results[~good][self.eval_columns].sort_values('idx'))

if __name__=='__main__':
    evaluator = EvaluateResults()
    evaluator.is_log_q_good()
    evaluator.make_all_scatter_plots()
    evaluator.make_all_delta_plots()
    #plt.show()
