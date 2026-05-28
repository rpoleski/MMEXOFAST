"""
Analyze a planet light curve from the 2018 Data Challenge. Minimal user effort.
"""
import os.path
import glob
import numpy as np
import traceback


from mmexofast.config import MODULE_PATH
import mmexofast as mmexo
from examples.DC18_classes import dir_, TestDataSet


base_dir = os.path.join(
            MODULE_PATH, 'DC18Test', 'temp_output', 'no_par')


def fit_lc(lc_num, verbose=False):
    data = TestDataSet(lc_num)
    output_dir = os.path.join(base_dir, 'W{0:03}'.format(lc_num))
    os.makedirs(output_dir, exist_ok=True)

    file_prefix = 'WFIRST.{0:03}'.format(lc_num)
    fitter = mmexo.MMEXOFASTFitter(
        files=[data.file_w149, data.file_z087], coords=data.coords, fit_type='binary_lens',
        verbose=verbose, renormalize_errors=False,
        no_parallax=True,
        log_file=os.path.join(output_dir, file_prefix + '.log'),
        restart_file=os.path.join(output_dir, file_prefix + '.pkl'),
        stop_after='fit_binary_lens:est_binary_params',
        limb_darkening_coeffs_gamma={'W149': 0., 'Z087': 0.},
        output_config=mmexo.OutputConfig(
            output_dir=output_dir, file_prefix=file_prefix, save_plots=True, save_table=True,
            save_exozippy_init=False)
    )
    fitter.fit()
    print(fitter.initialize_exozippy())
    #results = fitter.all_fit_results


def evaluate_results(lc_num):
    """
    Calculate metrics between input and output values
    Assume pymc output.
    """
    pass


files = glob.glob(os.path.join(dir_, 'n2018*.W149.*.txt'))
lc_nums = []
for file_ in files:
    elements = file_.split('.')
    lc_nums.append(int(elements[-2]))

# lc_nums for special cases
wide_planets = [8, 53, 107, 131, 152, 194, 208, 214, 217, 226]
big_wide_planets = [4, 62]
close_planets = [32, 40, 50, 74, 92, 95, 87,  186, 227]
big_close_planets = [27, 120, 124, 128, 172]
slow_parallax = [124, 128, 217] # 66 is broke
dip_anom = [47, 74, 95, 103]

lc_nums = [32]
for lc_num in np.sort(lc_nums):
    print('\n...Fitting light curve {0}...'.format(lc_num))
    try:
        results = fit_lc(lc_num, verbose=True)
        evaluate_results(lc_num)
    except NotImplementedError:
        pass
    except Exception as e:
        print('Run {0} ABORTED. {1}: {2}'.format(lc_num, type(e).__name__, e))
        traceback.print_exc()
